"""M2: encoder + per-point decoder (spec 7.3). M1: encoder + global pooling + MLP -> Cd (spec 7.4)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import Encoder


def global_vec(x):
    return torch.cat([x.mean((2, 3, 4)), x.amax((2, 3, 4))], 1)


class M2(nn.Module):
    """forward(vol, g, ndc, nrm) -> (B, N) normalised Cp. g: grid_sample coordinates (Z padding accounted for); ndc, nrm: point features.
    use_xyz / use_nrm = False for the M2-noXYZ / M2-noN ablations."""

    def __init__(self, chs=(16, 32, 64, 128, 128), in_ch=1, use_xyz=True, use_nrm=True, proj=32, glob=64, fourier=0, use_glob=True, n_local=0):
        super().__init__()
        self.enc, self.use_xyz, self.use_nrm, self.proj, self.fourier = Encoder(chs, in_ch), use_xyz, use_nrm, proj, fourier
        self.use_glob, self.n_local = use_glob, n_local  # use_glob=False: no global vector in the decoder; n_local: extra per-point features (src/data/local_features.py)
        self.reduce = nn.ModuleList([nn.Conv3d(c, proj, 1) for c in chs[1:]])
        if use_glob:
            self.glob = nn.Linear(2 * chs[-1], glob)
        n = proj * (len(chs) - 1) + glob * use_glob + 3 * use_xyz * (1 + 2 * fourier) + 3 * use_nrm + n_local
        self.mlp = nn.Sequential(nn.Linear(n, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, vol, g, ndc, nrm, loc=None):
        feats = self.enc(vol)
        B, N, _ = g.shape
        grid = g.float().view(B, N, 1, 1, 3)
        f = [F.grid_sample(r(x).float(), grid, mode="bilinear", padding_mode="border", align_corners=False).view(B, -1, N).transpose(1, 2)
             for r, x in zip(self.reduce, feats)]
        parts = f + ([self.glob(global_vec(feats[-1])).unsqueeze(1).expand(B, N, -1).to(f[0].dtype)] if self.use_glob else [])
        if self.use_xyz:
            x = ndc.float()
            if self.fourier:  # Fourier encoding of query coordinates (optional, outside the spec; off by default)
                fr = (2.0 ** torch.arange(self.fourier, device=x.device)) * torch.pi
                x = torch.cat([x] + [fn(x[..., None] * fr).flatten(-2) for fn in (torch.sin, torch.cos)], -1)
            parts.append(x.to(f[0].dtype))
        if self.use_nrm:
            parts.append(nrm.to(f[0].dtype))
        if self.n_local:
            parts.append(loc.to(f[0].dtype))
        return self.mlp(torch.cat(parts, -1)).squeeze(-1)


class M1(nn.Module):
    def __init__(self, chs=(16, 32, 64, 128, 128), in_ch=1):
        super().__init__()
        self.enc = Encoder(chs, in_ch)
        self.head = nn.Sequential(nn.Linear(2 * chs[-1], 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, vol, *_):
        return self.head(global_vec(self.enc(vol)[-1])).squeeze(-1)
