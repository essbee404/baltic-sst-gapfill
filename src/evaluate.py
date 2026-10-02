"""Evaluate the model and baselines on the test years: RMSE, CRPS, interval coverage and
marine heatwave (MHW) detection, separately for the training region and the held-out Gulf of Bothnia."""
import argparse
import json

import numpy as np
import torch
from scipy.special import erfinv
from scipy.stats import norm

import baselines
from data import Baltic, GapDataset, n_input_channels
from model import UNet
from utils import get_device, load_config, set_seed

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


# ---------------------------------------------------------------- probabilistic metrics

def crps_gaussian(mu, sigma, y):
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def coverage(mu, sigma, y, levels=LEVELS):
    """Fraction of truths inside the central interval of each nominal probability."""
    z = np.sqrt(2) * erfinv(levels)[:, None]
    return (np.abs(y - mu)[None] <= z * sigma[None]).mean(1)


# ---------------------------------------------------------------- marine heatwaves

def mhw_days(above, min_dur=5, max_gap=2):
    """Hobday et al. (2016): runs of >= min_dur days above the threshold are events; events
    separated by <= max_gap days are merged. above: (T, N) bool -> (T, N) bool MHW-day flags."""
    T, N = above.shape
    out = np.zeros_like(above)
    for j in range(N):
        a = np.concatenate([[False], above[:, j], [False]]).astype(int)
        starts, ends = np.where(np.diff(a) == 1)[0], np.where(np.diff(a) == -1)[0]
        keep = (ends - starts) >= min_dur
        starts, ends = starts[keep], ends[keep]
        if len(starts) == 0:
            continue
        merged = [[starts[0], ends[0]]]
        for s, e in zip(starts[1:], ends[1:]):
            if s - merged[-1][1] <= max_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            out[s:e, j] = True
    return out


def f1(pred, true):
    tp = (pred & true).sum()
    p, r = tp / max(pred.sum(), 1), tp / max(true.sum(), 1)
    return dict(f1=2 * p * r / max(p + r, 1e-9), precision=p, recall=r)


# ---------------------------------------------------------------- prediction

@torch.no_grad()
def predict(model, ds, k, bs=64):
    device = next(model.parameters()).device
    """Run every method over a dataset. Returns per-day full fields (T, H, W) for each method:
    {name: (mu, var or difficulty feature)}, plus truth, target mask and observed-today mask."""
    dl = torch.utils.data.DataLoader(ds, bs, num_workers=4)
    out = {"model": ([], []), "climatology": ([], [])}
    out.update({n: ([], []) for n in baselines.METHODS})
    ys, ms, obs_today, days = [], [], [], []
    for x, y, m, t in dl:
        mu, var = model(x.to(device))
        out["model"][0].append(mu.cpu().numpy())
        out["model"][1].append(var.cpu().numpy())
        xn = x.numpy()
        for name, fn in [("climatology", baselines.climatology)] + list(baselines.METHODS.items()):
            a, b = fn(xn, k)
            out[name][0].append(a.astype(np.float32))
            out[name][1].append(b.astype(np.float32))
        ys.append(y.numpy())
        ms.append(m.numpy())
        obs_today.append(xn[:, 3 * k + 1] > 0.5)  # observation mask of the target day
        days.append(t.numpy())
    cat = np.concatenate
    return ({n: (cat(a), cat(b)) for n, (a, b) in out.items()},
            cat(ys), cat(ms), cat(obs_today), cat(days))


def calibrate_baselines(preds, y, m):
    """Fit each baseline's variance on validation data (difficulty feature -> MSE)."""
    cal = {}
    for name in baselines.METHODS:
        mu, feat = preds[name]
        cal[name] = baselines.BinnedVariance().fit(feat[m], (mu[m] - y[m]) ** 2)
    return cal


def apply_calibration(preds, cal):
    for name, c in cal.items():
        mu, feat = preds[name]
        preds[name] = (mu, c(feat).astype(np.float32))
    return preds


def scores(preds, y, sel):
    rows = {}
    for name, (mu, var) in preds.items():
        mu_, s_, y_ = mu[sel], np.sqrt(var[sel]), y[sel]
        cov = coverage(mu_, s_, y_)
        rows[name] = dict(rmse=float(np.sqrt(((mu_ - y_) ** 2).mean())),
                          crps=float(crps_gaussian(mu_, s_, y_).mean()),
                          cov90=float(cov[LEVELS == 0.9][0]),
                          coverage=cov.tolist(), n=int(sel.sum()))
    return rows


def heatwave_scores(bal, preds, y, m, obs_today, days, region, cfg):
    """Reconstruct each day (observed pixels kept, hidden pixels predicted), detect MHWs on the
    reconstruction and on the truth, and compare MHW-day flags on the hidden pixel-days."""
    e = cfg["eval"]
    clim = bal.clim_mean[bal.doy[days] - 1]
    thr = bal.thresh[bal.doy[days] - 1]
    valid = bal.valid[days] & region[None] & np.isfinite(thr)
    cols = region & bal.ocean
    truth_sst = np.where(valid, y + clim, -np.inf)
    true_mhw = mhw_days((truth_sst > thr)[:, cols], e["mhw_min_duration"], e["mhw_max_gap"])
    hidden = (m & region[None])[:, cols]
    res, flags = {}, {"truth": true_mhw}
    for name, (mu, _) in preds.items():
        recon = np.where(obs_today, y, mu) + clim
        recon = np.where(valid, recon, -np.inf)
        pred_mhw = mhw_days((recon > thr)[:, cols], e["mhw_min_duration"], e["mhw_max_gap"])
        res[name] = {k: float(v) for k, v in f1(pred_mhw[hidden], true_mhw[hidden]).items()}
        flags[name] = pred_mhw
    return res, flags, true_mhw[hidden]


def main(cfg, ckpt="results/model.pt", out="results/metrics.json", frac=None):
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["train"]["num_threads"])
    bal = Baltic(cfg)
    s, k = cfg["split"], cfg["gaps"]["half_window"]
    model = UNet(n_input_channels(cfg), cfg["model"]["base_channels"], cfg["model"]["levels"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(get_device(cfg["train"]["device"]))
    model.eval()
    seed = cfg["eval"]["seed"]
    fr = [frac, frac] if frac is not None else None

    # Baseline variances are fitted on the validation years in the training region.
    val = GapDataset(bal, bal.day_indices(s["val_years"]), bal.regions["train"], seed=seed, frac_range=fr)
    vp, vy, vm, _, _ = predict(model, val, k)
    cal = calibrate_baselines(vp, vy, vm)

    # Test years: the model sees the whole Baltic as input (clouds permitting); metrics per region.
    test = GapDataset(bal, bal.day_indices(s["test_years"]), bal.ocean, seed=seed, frac_range=fr)
    preds, y, m, obs_today, days = predict(model, test, k)
    preds = apply_calibration(preds, cal)

    results = {}
    for rname in ("train", "test"):
        region = bal.regions[rname]
        sel = m & region[None]
        r = {"all_hidden": scores(preds, y, sel)}
        hw, flags, true_hidden = heatwave_scores(bal, preds, y, m, obs_today, days, region, cfg)
        # Same probabilistic scores restricted to pixel-days that are truly part of a MHW.
        mhw_full = np.zeros_like(m)
        mhw_full[:, region & bal.ocean] = flags["truth"]
        r["mhw_days"] = scores(preds, y, sel & mhw_full)
        r["mhw_detection"] = hw
        r["mhw_hidden_pixel_days"] = int(true_hidden.sum())
        results[rname] = r
        print(f"\n== {rname} region, test years ==")
        print(f"{'method':26s} {'RMSE':>6s} {'CRPS':>6s} {'cov90':>6s} {'RMSE@MHW':>9s} {'MHW F1':>7s}")
        for n in preds:
            a, h = r["all_hidden"][n], r["mhw_days"][n]
            print(f"{n:26s} {a['rmse']:6.3f} {a['crps']:6.3f} {a['cov90']:6.3f} "
                  f"{h['rmse']:9.3f} {hw[n]['f1']:7.3f}")

    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(out.replace(".json", "_fields.npz"), days=days, y=y, m=m, obs=obs_today,
                        **{f"mu_{n}": p[0] for n, p in preds.items()},
                        **{f"var_{n}": p[1] for n, p in preds.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default="results/model.pt")
    p.add_argument("--out", default="results/metrics.json")
    p.add_argument("--frac", type=float, default=None,
                   help="evaluate at a fixed gap fraction instead of the configured range")
    a = p.parse_args()
    main(load_config(a.config), a.ckpt, a.out, a.frac)
