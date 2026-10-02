"""Figures: reconstruction map with uncertainty, calibration curve, 2018 heatwave timeline."""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import Baltic  # noqa: E402
from evaluate import LEVELS, mhw_days  # noqa: E402
from utils import load_config  # noqa: E402

COLORS = {"model": "#2a6fdb", "Spatio-temporal interp.": "#e08a1e",
          "Linear interp. (time)": "#7a9a3a", "climatology": "#8c8c8c"}
LABELS = {"model": "U-Net (ours)", "climatology": "Climatology"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def crop(a, bal):
    return a[..., :bal.H, :bal.W]


def fig_map(bal, f, cfg):
    day = pd.Timestamp(cfg["eval"]["example_date"]) + pd.Timedelta(hours=0)
    t = int(np.where(bal.times == day)[0][0])
    i = int(np.where(f["days"] == t)[0][0])
    clim = bal.clim_mean[bal.doy[t] - 1]
    ocean = bal.valid[t]
    truth = np.where(ocean, f["y"][i] + clim, np.nan)
    obs = np.where(f["obs"][i] & ocean, truth, np.nan)
    recon = np.where(ocean, np.where(f["obs"][i], f["y"][i], f["mu_model"][i]) + clim, np.nan)
    sigma = np.where(ocean & ~f["obs"][i], np.sqrt(f["var_model"][i]), np.nan)
    err = np.where(f["m"][i], np.abs(f["mu_model"][i] - f["y"][i]), np.nan)

    panels = [("Truth (OISST)", truth, "RdYlBu_r"), ("Input: target day under clouds", obs, "RdYlBu_r"),
              ("Reconstruction (mean)", recon, "RdYlBu_r"), ("Predicted σ (°C)", sigma, "magma_r"),
              ("|error| on hidden pixels (°C)", err, "magma_r")]
    vmin, vmax = np.nanpercentile(truth, [2, 98])
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), constrained_layout=True)
    lon, lat = bal.lon, bal.lat
    for ax, (title, a, cmap) in zip(axes, panels):
        a = crop(a, bal)
        kw = dict(vmin=vmin, vmax=vmax) if cmap == "RdYlBu_r" else dict(vmin=0, vmax=1.5)
        im = ax.pcolormesh(lon, lat, a, cmap=cmap, shading="nearest", **kw)
        ax.set_facecolor("#d9d9d9")
        ax.contour(lon, lat, crop(bal.regions["test"] & bal.ocean, bal).astype(float), [0.5],
                   colors="k", linewidths=0.8, linestyles="--")
        ax.set_title(title)
        ax.set_aspect(1 / np.cos(np.deg2rad(60)))
        fig.colorbar(im, ax=ax, shrink=0.8, label="°C" if cmap == "RdYlBu_r" else None)
    fig.suptitle(f"{day:%d %b %Y}: Baltic SST reconstruction. Dashed: held-out Gulf of Bothnia "
                 "(never seen in training)")
    fig.savefig("figures/reconstruction_map.png", dpi=150)
    plt.close(fig)


def fig_calibration(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True, sharey=True)
    for ax, (rname, title) in zip(axes, [("train", "Training region"),
                                         ("test", "Held-out Gulf of Bothnia")]):
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        for n, r in metrics[rname]["all_hidden"].items():
            ax.plot(LEVELS, r["coverage"], "o-", ms=3, color=COLORS[n], label=LABELS.get(n, n))
        ax.set(title=f"{title}, 2017–2023", xlabel="Nominal central-interval probability",
               xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("Observed coverage")
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig("figures/calibration.png", dpi=150)
    plt.close(fig)


def fig_timeline(bal, f, cfg):
    e = cfg["eval"]
    days = f["days"]
    region = bal.regions["test"] & bal.ocean
    thr = bal.thresh[bal.doy[days] - 1]
    clim = bal.clim_mean[bal.doy[days] - 1]
    valid = bal.valid[days] & region[None] & np.isfinite(thr)

    def frac_mhw(sst):
        above = np.where(valid, sst, -np.inf) > thr
        flags = mhw_days(above[:, region], e["mhw_min_duration"], e["mhw_max_gap"])
        return flags.sum(1) / np.maximum(valid[:, region].sum(1), 1)

    t = bal.times[days]
    sel = (t >= "2018-05-01") & (t <= "2018-09-30")
    truth = f["y"] + clim
    # One representative pixel in the Bothnian Sea to show the per-pixel uncertainty band.
    pi, pj = np.abs(bal.lat - 62.1).argmin(), np.abs(bal.lon - 19.6).argmin()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, constrained_layout=True)
    obs = f["obs"][:, pi, pj]
    mu = np.where(obs, f["y"][:, pi, pj], f["mu_model"][:, pi, pj]) + clim[:, pi, pj]
    sd = np.where(obs, 0, np.sqrt(f["var_model"][:, pi, pj]))
    a1.fill_between(t[sel], (mu - 1.645 * sd)[sel], (mu + 1.645 * sd)[sel], color=COLORS["model"],
                    alpha=0.25, lw=0, label="U-Net 90% interval (hidden days)")
    a1.plot(t[sel], truth[sel, pi, pj], "k", lw=1.2, label="Truth")
    a1.plot(t[sel], mu[sel], color=COLORS["model"], lw=1, label="Reconstruction")
    a1.plot(t[sel], thr[sel, pi, pj], "r--", lw=1, label="MHW threshold (90th pct)")
    a1.plot(t[sel], clim[sel, pi, pj], color="0.5", lw=1, label="Climatology")
    a1.set(ylabel="SST (°C)",
           title=f"2018 Baltic heatwave at {bal.lat[pi]:.2f}°N {bal.lon[pj]:.2f}°E (held-out region)")
    a1.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")

    a2.plot(t[sel], frac_mhw(truth)[sel], "k", lw=1.5, label="Truth")
    for n in ["model", "Spatio-temporal interp.", "Linear interp. (time)"]:
        recon = np.where(f["obs"], f["y"], f[f"mu_{n}"]) + clim
        a2.plot(t[sel], frac_mhw(recon)[sel], color=COLORS[n], lw=1, label=LABELS.get(n, n))
    a2.set(ylabel="Fraction of region\nin a marine heatwave", ylim=(0, 1))
    a2.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig("figures/heatwave_2018.png", dpi=150)
    plt.close(fig)


def main(cfg):
    bal = Baltic(cfg)
    f = dict(np.load("results/metrics_fields.npz"))
    with open("results/metrics.json") as fh:
        metrics = json.load(fh)
    fig_map(bal, f, cfg)
    fig_calibration(metrics)
    fig_timeline(bal, f, cfg)
    print("figures written to figures/")


if __name__ == "__main__":
    main(load_config())
