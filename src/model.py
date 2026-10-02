"""Small U-Net with two output heads: mean and variance of the SST anomaly at every pixel."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, in_ch, base=32, levels=3):
        super().__init__()
        chs = [base * 2 ** i for i in range(levels + 1)]
        self.down = nn.ModuleList([block(in_ch, chs[0])] +
                                  [block(chs[i], chs[i + 1]) for i in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(levels))])
        self.dec = nn.ModuleList([block(2 * chs[i], chs[i]) for i in reversed(range(levels))])
        self.head = nn.Conv2d(chs[0], 2, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        mu, raw = self.head(x).unbind(1)
        var = F.softplus(raw) + 1e-3  # softplus keeps the variance positive without exp blow-ups
        return mu, var


def gaussian_nll(mu, var, y, mask):
    """Mean Gaussian negative log-likelihood over the masked (cloud-hidden, valid) pixels.
    Unlike MSE, this also trains the variance head: the model is penalised both for being
    wrong and for being over- or under-confident about how wrong it is."""
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    return nll[mask].mean() if mask.any() else nll.sum() * 0.0  # e.g. a fully ice-covered batch
