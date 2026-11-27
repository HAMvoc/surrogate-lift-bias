"""Trainer (spec 8): AdamW, warmup + cosine, gradient accumulation, EMA, AMP, stop/resumable (last.pt, best.pt, log.csv).
Car order and point samples are derived from (seed, epoch, step), so resuming mid-epoch at any point gives the same data sequence."""
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.metrics import car_stats


def git_commit():
    if os.environ.get("GIT_COMMIT"):  # Kaggle has no .git: the notebook sets this variable
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return ""


def cfg_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


class Trainer:
    """task 'cp': M2 predicts per-point Cp (Huber loss); task 'cd': M1 predicts Cd (MSE)."""

    def __init__(self, model, train, val, cfg, run_dir, device="cuda", seed=0, task="cp"):
        self.cfg, self.train, self.val, self.task, self.seed = cfg, train, val, task, seed
        self.dev = torch.device(device)
        torch.manual_seed(seed)
        self.model = model.to(self.dev)
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg["weight_decay"])
        self.steps_per_epoch = math.ceil(len(train) / (cfg["batch"] * cfg["accum"]))
        total, warm = cfg["epochs"] * self.steps_per_epoch, cfg["warmup_epochs"] * self.steps_per_epoch
        if cfg.get("step_epochs"):  # Transolver (as in CarBench): lr x gamma every step_epochs epochs, no warmup
            step = cfg["step_epochs"] * self.steps_per_epoch
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: cfg["gamma"] ** (s // step))
        else:
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: (s + 1) / warm if s < warm else
                                                           (cfg["min_lr"] / cfg["lr"] + (1 - cfg["min_lr"] / cfg["lr"]) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))
        self.amp = cfg["amp"]  # 'bf16' | 'fp16' | 'none'
        self.scaler = torch.amp.GradScaler(enabled=self.amp == "fp16" and self.dev.type == "cuda")
        y = train.cp if task == "cp" else train.cd
        self.mu, self.sd = float(y.mean()), float(y.std())  # label normalisation from train only (spec 6.2 no. 2)
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.epoch, self.best, self.hash = 0, float("inf"), cfg_hash(cfg)
        self.best_epoch = 0
        if (self.dir / "last.pt").exists():
            self.load(self.dir / "last.pt")

    # ---------- normalisation, forward ----------
    def _ctx(self):
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.amp)
        return torch.autocast(self.dev.type, dtype=dt, enabled=dt is not None and self.dev.type == "cuda")

    def _forward(self, model, b):
        b = {k: v.to(self.dev, non_blocking=True) for k, v in b.items()}
        with self._ctx():
            out = model(b["vol"], b["g"], b["ndc"], b["nrm"], *([b["loc"]] if "loc" in b else [])) if self.task == "cp" else model(b["vol"])
        return out.float(), b

    def _loss(self, out, b):
        if self.task == "cp":
            tgt = (b["cp"] - self.mu) / self.sd
            loss = F.huber_loss(out, tgt, delta=1.0)
            lam = self.cfg.get("lam_int")  # prereg addendum 3: per-car pressure-integral error (lift ~ n_z, drag ~ n_x)
            if lam:
                d = out - tgt
                loss = loss + lam * (((d * b["nrm"][..., 2]).mean(1) ** 2) + ((d * b["nrm"][..., 0]).mean(1) ** 2)).mean()
            return loss
        return F.mse_loss(out, (b["cd"] - self.mu) / self.sd)

    # ---------- training ----------
    def _gen(self, epoch, step):
        return torch.Generator().manual_seed(int(hashlib.md5(f"{self.seed}-{epoch}-{step}".encode()).hexdigest()[:8], 16))

    def train_epoch(self):
        c = self.cfg
        order = np.random.default_rng([self.seed, self.epoch]).permutation(len(self.train))
        per_step, losses, n_cars = c["batch"] * c["accum"], [], 0
        self.model.train()
        for s in range(self.steps_per_epoch):
            ids = order[s * per_step:(s + 1) * per_step]
            micro = [ids[i:i + c["batch"]] for i in range(0, len(ids), c["batch"])]
            for k, m in enumerate(micro):
                out, b = self._forward(self.model, self.train.batch(m, c["n_points"], self._gen(self.epoch, s * c["accum"] + k)) if self.task == "cp"
                                       else self.train.batch(m, 1, self._gen(self.epoch, s)))
                loss = self._loss(out, b)
                self.scaler.scale(loss * len(m) / len(ids)).backward()
                losses.append(loss.item())
                n_cars += len(m)
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["clip"])
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            g = self.epoch * self.steps_per_epoch + s
            d = min(c["ema"], (1 + g) / (10 + g))  # fast-start EMA
            with torch.no_grad():
                for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(self.ema.buffers(), self.model.buffers()):  # BatchNorm running stats (no-op for GroupNorm/LayerNorm models)
                    be.copy_(bm)
        return float(np.mean(losses)), n_cars

    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate(self, data, model=None, keep=False):
        """Per-car DataFrame. task cp: rel L2 on clipped and raw Cp, using the whole pool; task cd: Cd error."""
        model = model or self.ema
        model.eval()
        chunk = self.cfg.get("eval_chunk", 100000)
        rows = []
        self.preds = []  # keep=True: per-point predictions (fp16, in pool order) for region analysis and C_D,p
        for j, sid in enumerate(data.ids):
            if self.task == "cd":
                out, b = self._forward(model, data.batch([j], 1))
                p = out.item() * self.sd + self.mu
                rows.append({"id": sid, "cd": float(data.cd[j]), "pred": p, "abs_err": abs(p - float(data.cd[j]))})
                continue
            full = data.batch([j])
            preds = []
            feats_in = {k: v for k, v in full.items()}
            for a in range(0, full["g"].shape[1], chunk):
                part = {k: (v[:, a:a + chunk] if v.dim() > 1 and k in ("g", "ndc", "nrm", "cp", "cp_raw", "loc") else v) for k, v in feats_in.items()}
                out, _ = self._forward(model, part)
                preds.append(out[0].cpu() * self.sd + self.mu)
            p = torch.cat(preds).numpy()
            if keep:
                self.preds.append(p.astype(np.float16))
            c, r = car_stats(p, data.cp[j].numpy()), car_stats(p, data.cp_raw[j].numpy())
            rows.append({"id": sid, **c, **{f"raw_{k}": v for k, v in r.items() if k in ("rel_l2", "mae", "rmse", "sse", "sum_c", "sum_c2")}})
        return pd.DataFrame(rows)

    def val_metric(self, df):
        return float(df.rel_l2.mean()) if self.task == "cp" else float(df.abs_err.mean())

    # ---------- checkpoint ----------
    def save(self, name):
        torch.save({"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                    "scaler": self.scaler.state_dict(), "epoch": self.epoch, "best": self.best, "best_epoch": self.best_epoch, "mu": self.mu, "sd": self.sd,
                    "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                    "cfg": self.cfg, "cfg_hash": self.hash, "commit": git_commit()}, self.dir / name)

    def load(self, path):
        s = torch.load(path, map_location=self.dev, weights_only=False)
        assert s["cfg_hash"] == self.hash, f"config differs from checkpoint ({s['cfg_hash']} != {self.hash})"
        self.model.load_state_dict(s["model"])
        self.ema.load_state_dict(s["ema"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.scaler.load_state_dict(s["scaler"])
        self.epoch, self.best, self.mu, self.sd = s["epoch"], s["best"], s["mu"], s["sd"]
        self.best_epoch = s.get("best_epoch", 0)
        random.setstate(s["rng"]["py"])
        np.random.set_state(s["rng"]["np"])
        torch.set_rng_state(s["rng"]["torch"].cpu())
        if s["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in s["rng"]["cuda"]])

    # ---------- loop ----------
    def fit(self, max_seconds=None, log=print):
        t0, logf = time.time(), self.dir / "log.csv"
        while self.epoch < self.cfg["epochs"]:
            te = time.time()
            if self.dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loss, n = self.train_epoch()
            self.epoch += 1
            ve = self.cfg.get("val_every", 1)  # validate every ve epochs and at the last one (default: every epoch, as for all earlier runs)
            m = self.val_metric(self.evaluate(self.val)) if self.epoch % ve == 0 or self.epoch == self.cfg["epochs"] else float("nan")
            row = {"epoch": self.epoch, "train_loss": loss, "val": m, "lr": self.opt.param_groups[0]["lr"], "cars_per_s": n / (time.time() - te),
                   "max_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if self.dev.type == "cuda" else 0.0, "seconds": time.time() - te}
            pd.DataFrame([row]).to_csv(logf, mode="a", header=not logf.exists(), index=False)
            log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            if m < self.best:  # NaN (no validation this epoch) compares False
                self.best, self.best_epoch = m, self.epoch
                self.save("best.pt")
            self.save("last.pt")
            if self.cfg.get("patience") and self.epoch - self.best_epoch >= self.cfg["patience"]:  # early stopping (Transolver study, CarBench rule), counted in epochs
                log(f"early stop: no validation improvement since epoch {self.best_epoch}")
                return True
            if max_seconds and time.time() - t0 + (time.time() - te) > max_seconds:
                log("out of time: saved last.pt, exiting to resume in the next session")
                return False
        return True
