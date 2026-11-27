"""B4: Transolver (Physics-Attention) for per-point surface pressure.
Vendored from CarBench (Elrefaie et al., github.com/Mohamedelrefaie/CarBench, commit 66202db98e7564fb73367211cf81b278e83e6306, MIT licence),
models/transolver/model.py, itself the Physics_Attention_Irregular_Mesh block of THUML Transolver (github.com/thuml/Transolver, MIT).
Layers and forward pass unchanged (surface + normals input, 6 features, `unified_pos` off, `use_sdf` off). Differences, all interface or dependency only:
`einops.rearrange` -> reshape, `timm.layers.trunc_normal_` -> torch.nn.init.trunc_normal_, the 7-feature SDF mode and the lightweight class are dropped,
and `TransolverPC` adapts the call to our batch (forward(vol, g, ndc, nrm): the volume is not used). Config of CarBench models/transolver/config.yaml:
hidden 256, 5 blocks, 8 heads, 32 slices, mlp_ratio 2, dropout 0."""
import torch
import torch.nn as nn


class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner = dim_head * heads
        self.dim_head, self.heads, self.scale = dim_head, heads, dim_head ** -0.5
        self.softmax, self.dropout = nn.Softmax(dim=-1), nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_x, self.in_project_fx = nn.Linear(dim, inner), nn.Linear(dim, inner)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim_head, dim_head, bias=False) for _ in range(3))
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):  # (B, N, C)
        B, N, _ = x.shape
        fx = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
        xm = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        w = self.softmax(self.in_project_slice(xm) / self.temperature)  # B H N G
        tok = torch.einsum("bhnc,bhng->bhgc", fx, w) / (w.sum(2) + 1e-5)[..., None]  # slice tokens: weighted mean over ALL points of the input set
        q, k, v = self.to_q(tok), self.to_k(tok), self.to_v(tok)
        out = torch.matmul(self.dropout(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale)), v)
        out = torch.einsum("bhgc,bhng->bhnc", out, w).permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, heads, dim, dropout, mlp_ratio, last, slice_num):
        super().__init__()
        self.last = last
        self.ln_1, self.ln_2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, heads, dim // heads, dropout, slice_num)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        if last:
            self.ln_3, self.out = nn.LayerNorm(dim), nn.Linear(dim, 1)

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return self.out(self.ln_3(fx)) if self.last else fx


class Transolver(nn.Module):
    def __init__(self, hidden=256, layers=5, heads=8, mlp_ratio=2, slice_num=32, dropout=0.0, in_dim=6):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(in_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.blocks = nn.ModuleList([Block(heads, hidden, dropout, mlp_ratio, i == layers - 1, slice_num) for i in range(layers)])
        self.placeholder = nn.Parameter((1 / hidden) * torch.rand(hidden))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features):  # (B, N, 6) = [x y z nx ny nz] -> (B, N)
        fx = self.pre(features) + self.placeholder[None, None]
        for b in self.blocks:
            fx = b(fx)
        return fx.squeeze(-1)


class TransolverPC(Transolver):
    """Same network, called as the other models of this repo: coordinates in the normalised box (as M2), unit normals; no volume, no global vector."""

    def forward(self, vol, g, ndc, nrm):
        return super().forward(torch.cat([ndc.float(), nrm.float()], -1))
