"""Transolver study, task T1: check that src/models/transolver.py equals the CarBench Transolver (same state dict after renaming, same output). Usage: python scripts/training/check_transolver_code.py <CarBench>/models/transolver/model.py"""
import sys, types, importlib.util, torch
# stubs for the two dependencies we dropped
ein = types.ModuleType("einops"); ein.rearrange = lambda x, p: x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
timm = types.ModuleType("timm"); tl = types.ModuleType("timm.layers"); tl.trunc_normal_ = torch.nn.init.trunc_normal_
sys.modules.update({"einops": ein, "timm": timm, "timm.layers": tl})
spec = importlib.util.spec_from_file_location("cbm", sys.argv[1]); cbm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cbm)
sys.path.insert(0, "."); from src.models.transolver import TransolverPC
ref = cbm.TransolverPressure({"n_hidden": 256, "n_layers": 5, "n_head": 8, "mlp_ratio": 2, "slice_num": 32, "dropout": 0.0, "use_sdf": False}).eval()
mine = TransolverPC().eval()
rn = lambda k: (k.replace("Attn.", "attn.").replace("mlp.linear_pre.0", "mlp.0").replace("mlp.linear_post", "mlp.2").replace("mlp2", "out").replace("preprocess.linear_pre.0", "pre.0").replace("preprocess.linear_post", "pre.2"))
sd = {rn(k): v for k, v in ref.state_dict().items()}
print("params ref", sum(p.numel() for p in ref.parameters()), "mine", sum(p.numel() for p in mine.parameters()))
print(mine.load_state_dict(sd, strict=True))
x = torch.randn(2, 3000, 6)
with torch.no_grad():
    print("max |ref - mine|", (ref(x) - mine(None, None, x[..., :3], x[..., 3:])).abs().max().item())
