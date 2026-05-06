import numpy as np

from ou_skew.models.ou import fit_ou_ar1, simulate_ou


def test_fit_ou_ar1_recovers_mean_reversion_direction():
    x = simulate_ou(theta=0.08, kappa=5.0, sigma=0.10, x0=0.02, n_steps=1500, seed=7)
    fit = fit_ou_ar1(x, min_obs=100)

    assert 0 < fit.phi < 1
    assert fit.kappa > 0
    assert np.isfinite(fit.theta)
    assert np.isfinite(fit.sigma)
    assert fit.stationary_std > 0
    assert fit.half_life_days > 0


def test_z_score_uses_stationary_std():
    x = simulate_ou(theta=0.05, kappa=4.0, sigma=0.08, x0=0.05, n_steps=800, seed=11)
    fit = fit_ou_ar1(x, min_obs=100)
    z = fit.z_score(fit.theta + fit.stationary_std)
    assert abs(z - 1.0) < 1e-8


def test_proxy_handles_negative_phi_series():
    from ou_skew.models.ou import fit_ou_or_stationary_proxy

    x = np.array([1.0, -0.35, 0.40, -0.15, 0.18, -0.05, 0.08, -0.01] * 10)
    fit = fit_ou_or_stationary_proxy(x, min_obs=20)

    assert fit.method in {"stationary_ar1_abs_phi_proxy", "sample_moment_proxy"}
    assert np.isfinite(fit.theta)
    assert fit.kappa > 0
    assert fit.stationary_std > 0
    assert fit.warning
