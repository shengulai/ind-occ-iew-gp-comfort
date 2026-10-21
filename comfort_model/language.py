"""Optional GPT-4o severity interpretation; importing this module makes no API calls."""
import ast
from dataclasses import dataclass
from importlib.resources import files
import math
import re


@dataclass(frozen=True)
class Interpretation:
    severity: float
    comfort: float
    response: str
    model: str


def parse_severity(response):
    candidates = []
    for block in re.findall(r'\[[^\[\]]*\]', response):
        try:
            values = ast.literal_eval(block)
        except (ValueError, SyntaxError):
            continue
        if values and all(isinstance(v, str) for v in values):
            continue  # Some replies echo the input text list.
        if len(values) != 1 or isinstance(values[0], bool) or not isinstance(values[0], (int, float)):
            raise ValueError('Expected exactly one numerical severity value.')
        value = float(values[0])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Severity must be finite and in [0, 1].')
        candidates.append(value)
    if len(candidates) != 1:
        raise ValueError('Missing or ambiguous severity in model response; no score was substituted.')
    return candidates[0]


def interpret_feedback(text, *, examples='', model='gpt-4o', temperature=.7, client=None):
    """Send one comment to the API and return severity plus comfort = 1 - severity.

    OPENAI_API_KEY is read by the official SDK. Environment readings are paired
    with the returned comfort by the caller; they are not part of this prompt.
    Pass per-person language examples to reproduce persona prompting.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Provide a nonempty feedback comment.')
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    system_prompt = files('comfort_model').joinpath('prompts/severity.txt').read_text()
    prompt = f"""
        Context: {system_prompt}
        User feedback examples: {examples}
        User feedback: {[text]}
    """
    completion = client.chat.completions.create(
        model=model, temperature=temperature, messages=[{'role': 'user', 'content': prompt}])
    response = completion.choices[0].message.content
    if not isinstance(response, str):
        raise ValueError('The API returned no textual interpretation.')
    severity = parse_severity(response)
    comfort = round(100 * (1 - severity), 2) / 100
    return Interpretation(severity, comfort, response, model)
