"""Train the U-Net on the training region and years; keep the checkpoint with the best validation NLL."""
import argparse
import json
import time

import torch

from data import Baltic, GapDataset, n_input_channels
from model import UNet, gaussian_nll
from utils import get_device, load_config, set_seed


def run_epoch(model, loader, device, opt=None, sched=None):
    model.train(opt is not None)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y, m, _ in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            mu, var = model(x)
            loss = gaussian_nll(mu, var, y, m)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()  # OneCycle is stepped per batch
            tot += loss.item() * len(x)
            n += len(x)
    return tot / n


def main(cfg, out="results/model.pt"):
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["train"]["num_threads"])
    bal = Baltic(cfg)
    s, tr = cfg["split"], cfg["train"]
    region = bal.regions["train"]  # Bothnia and the buffer zone are never shown to the model
    train_ds = GapDataset(bal, bal.day_indices(s["train_years"]), region)
    val_ds = GapDataset(bal, bal.day_indices(s["val_years"]), region, seed=cfg["eval"]["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = torch.utils.data.DataLoader(train_ds, tr["batch_size"], shuffle=True, generator=g,
                                           num_workers=4, persistent_workers=True)
    val_dl = torch.utils.data.DataLoader(val_ds, 64, num_workers=4, persistent_workers=True)

    device = get_device(tr["device"])
    model = UNet(n_input_channels(cfg), cfg["model"]["base_channels"], cfg["model"]["levels"]).to(device)
    print(f"device: {device}, ", end="")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, "
          f"{len(train_ds)} train days, {len(val_ds)} val days", flush=True)
    opt = torch.optim.AdamW(model.parameters(), tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tr["lr"], epochs=tr["epochs"],
                                                steps_per_epoch=len(train_dl))

    best, history = float("inf"), []
    for ep in range(tr["epochs"]):
        t0 = time.time()
        tl = run_epoch(model, train_dl, device, opt, sched)
        vl = run_epoch(model, val_dl, device)
        history.append(dict(epoch=ep, train_nll=tl, val_nll=vl))
        print(f"epoch {ep:2d}  train NLL {tl:.4f}  val NLL {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if vl < best:
            best = vl
            torch.save(model.state_dict(), out)
    with open(out.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="results/model.pt")
    a = p.parse_args()
    main(load_config(a.config), a.out)
