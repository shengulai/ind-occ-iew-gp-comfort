"""Numerical settings for the archived paper experiments; no machine paths."""
PERSONAS = ("Adam", "Bill", "Eve")
PMV = {
    'package_version': '4.6.2',
    'reference_options': dict(vr=0.1, rh=50.0, met=1.1, wme=0.0,
                              model='7730-2025', units='SI', round_output=False),
    'clothing_bounds': (0.3, 1.5),
    'temperature_range_F': (66.0, 80.0),
    'grid_size': 501,
    'training': dict(steps=2000, learning_rate=0.001, tau=120.0,
                     base_variance=0.005, history_variance=0.15, seed=0),
}

INITIAL = {'Adam': {'weights_initial': {'temp': 1, 'co2': 1},
          'wellness_range_initial': {'temp': [71, 77, 60, 85],
                                     'co2': [0, 900, 0, 1800]},
          'learn_weights_ini': False,
          'fixed_lower_bounds_ini': {'co2': 0.0},
          'normalize_inputs': True,
          'occ_rate': 1.0,
          'signal_var_low': 0.001,
          'signal_var_high': 0.1,
          'lengthscale_low': 0.02,
          'lengthscale_high': 0.2,
          'lengthscale_init': 0.1,
          'noise_llm_low': 0.001,
          'noise_llm_high': 0.05},
 'Bill': {'weights_initial': {'temp': 1, 'co2': 1},
          'wellness_range_initial': {'temp': [71, 77, 60, 85],
                                     'co2': [0, 900, 0, 1800]},
          'learn_weights_ini': False,
          'fixed_lower_bounds_ini': {'co2': 0.0},
          'normalize_inputs': True,
          'occ_rate': 1.0,
          'signal_var_low': 0.001,
          'signal_var_high': 0.1,
          'lengthscale_low': 0.02,
          'lengthscale_high': 0.2,
          'lengthscale_init': 0.1,
          'noise_llm_low': 0.001,
          'noise_llm_high': 0.05},
 'Eve': {'weights_initial': {'temp': 1, 'co2': 1},
         'wellness_range_initial': {'temp': [71, 74, 60, 85],
                                    'co2': [0, 900, 0, 1800]},
         'learn_weights_ini': False,
         'fixed_lower_bounds_ini': {'co2': 0.0},
         'normalize_inputs': True,
         'occ_rate': 1.0,
         'signal_var_low': 0.001,
         'signal_var_high': 0.1,
         'lengthscale_low': 0.02,
         'lengthscale_high': 0.2,
         'lengthscale_init': 0.1,
         'noise_llm_low': 0.001,
         'noise_llm_high': 0.05}}

REFERENCES = {'Adam': {'weights': {'temp': 1, 'co2': 1},
          'wellness_range': {'temp': [70.0, 74.0, 60.0, 85.0],
                             'co2': [0.0, 1000.0, 0.0, 1800.0]},
          'occ_rate': 1.0,
          'gp_params': {'noise_var': 0.005,
                        'signal_var': 0.004,
                        'lengthscale_temp': 1.0,
                        'lengthscale_co2': 100.0},
          'feedback_params': {'complaint_threshold': 0.85,
                              'p_at_threshold': 0.05,
                              'p_max': 0.3,
                              'p_min': 0.001,
                              'decay_rate': 15}},
 'Bill': {'weights': {'temp': 1, 'co2': 1},
          'wellness_range': {'temp': [74.0, 77.0, 60.0, 85.0],
                             'co2': [0.0, 700.0, 0.0, 1800.0]},
          'occ_rate': 1.0,
          'gp_params': {'noise_var': 0.006,
                        'signal_var': 0.004,
                        'lengthscale_temp': 1.5,
                        'lengthscale_co2': 150.0},
          'feedback_params': {'complaint_threshold': 0.8,
                              'p_at_threshold': 0.05,
                              'p_max': 0.25,
                              'p_min': 0.001,
                              'decay_rate': 25}},
 'Eve': {'weights': {'temp': 1, 'co2': 1},
         'wellness_range': {'temp': [72.0, 76.0, 60.0, 85.0],
                            'co2': [0.0, 1000.0, 0.0, 1800.0]},
         'occ_rate': 1.0,
         'gp_params': {'noise_var': 0.005,
                       'signal_var': 0.005,
                       'lengthscale_temp': 2.0,
                       'lengthscale_co2': 200.0},
         'feedback_params': {'complaint_threshold': 0.7,
                             'p_at_threshold': 0.02,
                             'p_max': 0.25,
                             'p_min': 0.001,
                             'decay_rate': 25}}}

TEMPORAL = {'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 'scenarios': ['stable', 'gradual', 'abrupt'],
 'taus': [50, 100, 200, 400],
 'history_variances': [0.15],
 'days': 720,
 'change_day': 300,
 'drift_duration': 120,
 'temp_shift': 2.0,
 'co2_shift': -200.0,
 'feedback_probability': 0.12,
 'reporting': 'uniform',
 'observation_sd': 0.07,
 'residual_sd': 0.025,
 'eval_every': 30,
 'grid_size': 20,
 'steps': 1000,
 'learning_rate': 0.001,
 'base_variance': 0.005,
 'global_noise_low': 0.001,
 'global_noise_high': 0.05,
 'global_noise_init': 0.005,
 'recovery_rmse': 0.08,
 'recovery_checkpoints': 3,
 'threads': 1}

ADVERSARIAL = {'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 'taus': [50, 100, 200, 400],
 'days': 720,
 'steps': 1000,
 'grid_size': 20,
 'eval_every': 30,
 'history_variances': [0.15],
 'feedback_probability': 0.12,
 'reporting': 'uniform',
 'observation_sd': 0.07,
 'residual_sd': 0.025,
 'learning_rate': 0.001,
 'base_variance': 0.005,
 'global_noise_low': 0.001,
 'global_noise_high': 0.05,
 'global_noise_init': 0.005,
 'threads': 1,
 'attack_start': 300,
 'attack_window': 60,
 'attack_count': 3,
 'event_eval_every': 10,
 'error_mode': 'opposite_endpoint',
 'recovery_tolerance': 0.01,
 'recovery_checkpoints': 3,
 'quick': False,
 'temp_shift': 0.0,
 'co2_shift': 0.0}

# Synthetic language generation definitions (not real people).
GENERATION = {'Adam': {'parameters': {'weights': {'temp': 1, 'co2': 1},
                         'wellness_range': {'temp': [70.0, 74.0, 60.0, 85.0],
                                            'co2': [0.0, 1000.0, 0.0, 1800.0]},
                         'occ_rate': 1.0,
                         'gp_params': {'noise_var': 0.005,
                                       'signal_var': 0.004,
                                       'lengthscale_temp': 1.0,
                                       'lengthscale_co2': 100.0},
                         'feedback_params': {'complaint_threshold': 0.85,
                                             'p_at_threshold': 0.05,
                                             'p_max': 0.3,
                                             'p_min': 0.001,
                                             'decay_rate': 15}},
          'character_info': {'description': '\n'
                                            '    Expertise about building '
                                            'science and comfort: low\n'
                                            '    Personality: you are a '
                                            'talkive and friendly person\n'
                                            '    Verbosity: you tend to talk '
                                            'long and in detail\n'
                                            '    Tone: you act politely and '
                                            'sometimes with a joke\n'
                                            '    Example feedback: "Well, it '
                                            'seems like the room is trying to '
                                            'simulate a sauna experience '
                                            'today, as it feels warmer than '
                                            "I'd prefer. Could we dial back "
                                            'the heat a notch, please?": '
                                            "'correspond comfort score': "
                                            '0.7041; "Well, I must say, the '
                                            'air in here is starting to feel '
                                            'more like a stuffy sauna than a '
                                            'sophisticated lab—perhaps a tad '
                                            'too warm and lacking in the fresh '
                                            "air department, don't you "
                                            'think?": \'correspond comfort '
                                            'score\': 0.2877; "Well, it seems '
                                            "like we're hosting a tropical "
                                            'vacation right here in the lab, '
                                            "doesn't it? It's feeling quite "
                                            'toasty in here, and not in the '
                                            'cozy way.": \'correspond comfort '
                                            "score': 0.6922\n"
                                            '    ',
                             'TRUE_PARAMS': {'weights': {'temp': 1, 'co2': 1},
                                             'wellness_range': {'temp': [70.0,
                                                                         74.0,
                                                                         60.0,
                                                                         85.0],
                                                                'co2': [0.0,
                                                                        1000.0,
                                                                        0.0,
                                                                        1800.0]},
                                             'occ_rate': 1.0,
                                             'gp_params': {'noise_var': 0.005,
                                                           'signal_var': 0.004,
                                                           'lengthscale_temp': 1.0,
                                                           'lengthscale_co2': 100.0},
                                             'feedback_params': {'complaint_threshold': 0.85,
                                                                 'p_at_threshold': 0.05,
                                                                 'p_max': 0.3,
                                                                 'p_min': 0.001,
                                                                 'decay_rate': 15}}}},
 'Bill': {'parameters': {'weights': {'temp': 1, 'co2': 1},
                         'wellness_range': {'temp': [74.0, 77.0, 60.0, 85.0],
                                            'co2': [0.0, 700.0, 0.0, 1800.0]},
                         'occ_rate': 1.0,
                         'gp_params': {'noise_var': 0.006,
                                       'signal_var': 0.004,
                                       'lengthscale_temp': 1.5,
                                       'lengthscale_co2': 150.0},
                         'feedback_params': {'complaint_threshold': 0.8,
                                             'p_at_threshold': 0.05,
                                             'p_max': 0.25,
                                             'p_min': 0.001,
                                             'decay_rate': 25}},
          'character_info': {'description': '\n'
                                            '    Expertise about building '
                                            'science and comfort: medium\n'
                                            '    Personality: you are very '
                                            'friendly and nice person, '
                                            'interested in helping this '
                                            'project\n'
                                            '    Verbosity: you talk very '
                                            'concisely\n'
                                            '    Tone: you talk directly to '
                                            'the point\n'
                                            '    Example feedback: None\n'
                                            '    ',
                             'TRUE_PARAMS': {'weights': {'temp': 1, 'co2': 1},
                                             'wellness_range': {'temp': [74.0,
                                                                         77.0,
                                                                         60.0,
                                                                         85.0],
                                                                'co2': [0.0,
                                                                        700.0,
                                                                        0.0,
                                                                        1800.0]},
                                             'occ_rate': 1.0,
                                             'gp_params': {'noise_var': 0.006,
                                                           'signal_var': 0.004,
                                                           'lengthscale_temp': 1.5,
                                                           'lengthscale_co2': 150.0},
                                             'feedback_params': {'complaint_threshold': 0.8,
                                                                 'p_at_threshold': 0.05,
                                                                 'p_max': 0.25,
                                                                 'p_min': 0.001,
                                                                 'decay_rate': 25}}}},
 'Eve': {'parameters': {'weights': {'temp': 1, 'co2': 1},
                        'wellness_range': {'temp': [72.0, 76.0, 60.0, 85.0],
                                           'co2': [0.0, 1000.0, 0.0, 1800.0]},
                        'occ_rate': 1.0,
                        'gp_params': {'noise_var': 0.005,
                                      'signal_var': 0.005,
                                      'lengthscale_temp': 2.0,
                                      'lengthscale_co2': 200.0},
                        'feedback_params': {'complaint_threshold': 0.7,
                                            'p_at_threshold': 0.02,
                                            'p_max': 0.25,
                                            'p_min': 0.001,
                                            'decay_rate': 25}},
         'character_info': {'description': '\n'
                                           '    Expertise about building '
                                           'science and comfort: high\n'
                                           '    Personality: you are very '
                                           'friendly and cool\n'
                                           '    Verbosity: you talk simple and '
                                           'concisely\n'
                                           '    Tone: you talk straight to the '
                                           'point and is polite\n'
                                           '    Example feedback: None\n'
                                           '    ',
                            'TRUE_PARAMS': {'weights': {'temp': 1, 'co2': 1},
                                            'wellness_range': {'temp': [72.0,
                                                                        76.0,
                                                                        60.0,
                                                                        85.0],
                                                               'co2': [0.0,
                                                                       1000.0,
                                                                       0.0,
                                                                       1800.0]},
                                            'occ_rate': 1.0,
                                            'gp_params': {'noise_var': 0.005,
                                                          'signal_var': 0.005,
                                                          'lengthscale_temp': 2.0,
                                                          'lengthscale_co2': 200.0},
                                            'feedback_params': {'complaint_threshold': 0.7,
                                                                'p_at_threshold': 0.02,
                                                                'p_max': 0.25,
                                                                'p_min': 0.001,
                                                                'decay_rate': 25}}}}}
