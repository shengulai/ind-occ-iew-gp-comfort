| Persona | Observations | Initial MAE | Learned MAE | Initial RMSE | Learned RMSE | Initial NLPD | Learned NLPD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adam | 60 | 0.1275 | 0.0405 | 0.2096 | 0.0678 | 2.6617 | -1.4504 |
| Bill | 23 | 0.0922 | 0.0918 | 0.1384 | 0.1548 | 0.1854 | -0.3068 |
| Eve | 8 | 0.2092 | 0.1823 | 0.3090 | 0.2748 | 7.8182 | 2.4069 |

NLPD is the mean marginal Gaussian negative log predictive density against the noise-free PMV-derived reference, using raw means and latent variances. No test observation noise is added and no clipping or variance recalibration is applied. The initial GP is unconditioned with signal variance 0.005; the learned GP uses its posterior latent variance. Lower NLPD is better; negative values are valid for continuous densities. NLPD alone does not establish interval calibration.
