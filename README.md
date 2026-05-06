# Ornstein-Uhlenbeck-Risk-Reversal

A Python research framework for modeling mean reversion in equity-option implied-volatility skew features using an Ornstein-Uhlenbeck (OU) process.

Real extracted skew series can be noisy; if the slope is negative or otherwise not
OU-admissible, plotting/simulation/z-score commands now fall back to a clearly
flagged stationary proxy instead of crashing. The output `status`/`method` field
will show whether the result is a strict `ou_ar1` fit or a proxy fit.
