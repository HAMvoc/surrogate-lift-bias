"""B2: PointNet++ lite (spec 7.4), pure PyTorch, per-point segmentation. Takes 16,384 points (coordinates + normals).
shortcut: centres are sampled randomly (not FPS) and grouped by kNN (not ball query), to run fast without external CUDA code; state this when reporting."""
import torch
import torch.nn as nn


def mlp(dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
    return nn.Sequential(*layers)


def knn(q, p, k):
    d = torch.cdist(q.float(), p.float())
    d, i = d.topk(min(k, p.shape[1]), dim=-1, largest=False)
    return d, i


def gather(x, idx):  # x (B,N,C), idx (B,M,k) -> (B,M,k,C)
    return x[torch.arange(x.shape[0], device=x.device)[:, None, None], idx]


class SA(nn.Module):
    def __init__(self, m, k, cin, dims):
        super().__init__()
        self.m, self.k, self.net = m, k, mlp([cin + 3] + dims)

    def forward(self, xyz, f):
        sel = torch.randperm(xyz.shape[1], device=xyz.device)[: self.m]
        cx = xyz[:, sel]
        _, idx = knn(cx, xyz, self.k)
        g = torch.cat([gather(xyz, idx) - cx[:, :, None], gather(f, idx)], -1)
        return cx, self.net(g).amax(2)


def interp(dst, src, f):
    d, i = knn(dst, src, 3)
    w = 1.0 / (d + 1e-8)
    return (gather(f, i) * (w / w.sum(-1, keepdim=True)).unsqueeze(-1)).sum(2)


class PointNet2Lite(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1, self.sa2 = SA(2048, 32, 6, [64, 64, 128]), SA(512, 32, 128, [128, 128, 256])
        self.glob = mlp([256 + 3, 256, 512])
        self.fp3, self.fp2, self.fp1 = mlp([512 + 256, 256]), mlp([256 + 128, 128]), mlp([128 + 6, 128, 128])
        self.out = nn.Linear(128, 1)

    def forward(self, vol, g, ndc, nrm):
        f0 = torch.cat([ndc, nrm], -1).float()
        x1, f1 = self.sa1(ndc.float(), f0)
        x2, f2 = self.sa2(x1, f1)
        gl = self.glob(torch.cat([x2, f2], -1)).amax(1, keepdim=True)
        h = self.fp3(torch.cat([f2, gl.expand(-1, f2.shape[1], -1)], -1))
        h = self.fp2(torch.cat([interp(x1, x2, h), f1], -1))
        h = self.fp1(torch.cat([interp(ndc.float(), x1, h), f0], -1))
        return self.out(h).squeeze(-1)
