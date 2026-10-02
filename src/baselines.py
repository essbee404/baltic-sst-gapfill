"""Baselines. Each returns a mean anomaly and a scalar 'difficulty' feature per pixel; the
predictive variance is then calibrated on validation data as the mean squared error per
difficulty bin, so every baseline is a fair probabilistic competitor, not just a point estimate.

Input x is the model's input array (B, C, H, W): channels [0, 2k] are observed anomalies,
[2k+1, 4k+1] the matching observation masks, and the last channel the climatological std.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def split_input(x, k):
    n = 2 * k + 1
    return x[:, :n], x[:, n:2 * n] > 0.5


def climatology(x, k):
    """Predict zero anomaly; spread = interannual std of SST for that day of year and pixel."""
    clim_std = x[:, -1]
    return np.zeros_like(clim_std), np.maximum(clim_std, 0.1) ** 2


def linear_interp(x, k):
    """Per pixel, linearly interpolate between the nearest observed day before and after the
    target day. One-sided -> persistence; no observation in the window -> climatology (0)."""
    vals, obs = split_input(x, k)
    B, _, H, W = vals.shape
    big = 99
    before_v, before_d = np.zeros((B, H, W)), np.full((B, H, W), big)
    after_v, after_d = np.zeros((B, H, W)), np.full((B, H, W), big)
    for d in range(k, 0, -1):  # walk inwards so the nearest observation wins
        o = obs[:, k - d]
        before_v[o], before_d[o] = vals[:, k - d][o], d
        o = obs[:, k + d]
        after_v[o], after_d[o] = vals[:, k + d][o], d
    has_b, has_a = before_d < big, after_d < big
    w = before_d / np.maximum(before_d + after_d, 1)
    mu = np.where(has_b & has_a, (1 - w) * before_v + w * after_v,
                  np.where(has_b, before_v, np.where(has_a, after_v, 0.0)))
    # Difficulty: total span of the interpolation; one-sided and no-data cases get their own bins.
    feat = np.where(has_b & has_a, before_d + after_d,
                    np.where(has_b | has_a, 10 + np.minimum(before_d, after_d), big))
    return mu, feat


def spatiotemporal(x, k, sigma_space=2.0, sigma_time=1.5):
    """Normalised convolution: Gaussian-weighted average of all observations in space and time,
    divided by the same weighting of the observation mask. A cheap stand-in for kriging."""
    vals, obs = split_input(x, k)
    obs = obs.astype(np.float64)
    wt = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma_time) ** 2)[None, :, None, None]
    s = (0, 0, sigma_space, sigma_space)
    num = (gaussian_filter(vals * obs, s, mode="constant") * wt).sum(1)
    den = (gaussian_filter(obs, s, mode="constant") * wt).sum(1)
    mu = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    return mu, -den  # little nearby data -> hard


class BinnedVariance:
    """Variance = validation MSE within quantile bins of a difficulty feature."""

    def __init__(self, n_bins=20):
        self.n_bins = n_bins

    def fit(self, feat, sqerr):
        self.edges = np.unique(np.quantile(feat, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        b = np.digitize(feat, self.edges)
        self.var = np.array([sqerr[b == i].mean() if (b == i).any() else sqerr.mean()
                             for i in range(len(self.edges) + 1)])
        return self

    def __call__(self, feat):
        return self.var[np.digitize(feat, self.edges)]


METHODS = {"Linear interp. (time)": linear_interp, "Spatio-temporal interp.": spatiotemporal}
