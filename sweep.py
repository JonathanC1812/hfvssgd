import time
import csv
import warnings

import numpy as np

from core import make_data, loss, optimal
from opt import gd, sgd, hf

warnings.filterwarnings("ignore")

#knobs
N, D = 100, 20
KAPPAS = np.logspace(0, 4, 9)
NOISES = [0.0, 0.1]
TOL = 1e-4
BATCH = 10
MAX_STEPS = 500_000
CG_TOL = 1e-6
SGD_LR_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5]
OUT = "results.csv"

def eigen_range(X):
    n = X.shape[0]
    ev = np.linalg.eigvalsh((X.T @ X) / n)
    return ev[0], ev[-1] #lambda_min, lambda_max

def timed(fn):
    t0 = time.perf_counter()
    w, st = fn()
    st["seconds"] = time.perf_counter() - t0
    return w, st

def tune_sgd_lr(X, y, L_star, lmax, mu): #for finding the learning rate of sgd
    best = None
    for c in SGD_LR_GRID:
        lr = c / lmax
        w, st = sgd(X, y, lr, batch=BATCH, tol=TOL, L_star=L_star,
                    max_steps=MAX_STEPS, mu=mu)
        key = (not st["converged"], st["cost"] if st["converged"] else loss(w, X, y))
        if best is None or key < best[0]:
            best = (key, lr)
    return best[1]

def run():
    rows = []
    for noise in NOISES:
        for kappa in KAPPAS:
            X, y, w_true = make_data(N, D, kappa, noise=noise)
            w_star, L_star = optimal(X, y)
            lmin, lmax = eigen_range(X)
            lr_gd = 2.0 / (lmin + lmax)
            mu = None if noise == 0.0 else lmin
            lr_sgd = tune_sgd_lr(X, y, L_star, lmax, mu)

            arms = {
                "HF": (lambda: hf(X, y, tol=TOL, L_star=L_star, cg_tol=CG_TOL), np.nan),
                "GD": (lambda: gd(X, y, lr_gd, tol=TOL, L_star=L_star,
                                  max_steps=MAX_STEPS), lr_gd),
                "SGD": (lambda: sgd(X, y, lr_sgd, batch=BATCH, tol=TOL, L_star=L_star,
                                    max_steps=MAX_STEPS, mu=mu), lr_sgd),
            }
            for name, (fn, lr) in arms.items():
                w, st = timed(fn)
                rows.append({
                    "noise": noise, "kappa": kappa, "method": name,
                    "lr": lr, "cost": st["cost"], "n_grad": st["n_grad"],
                    "n_hvp": st["n_hvp"], "steps": st["steps"],
                    "converged": int(st["converged"]), "seconds": st["seconds"],
                    "final_loss": loss(w, X, y), "L_star": L_star,
                    "w_err": np.linalg.norm(w - w_star) / max(np.linalg.norm(w_star), 1e-300),
                })
                print(f"noise={noise} kappa={kappa:>9.1f} {name:>3} "
                      f"cost={st['cost']:>12.1f} grad={st['n_grad']:>10.1f} "
                      f"hvp={st['n_hvp']:>5.0f} conv={int(st['converged'])} "
                      f"{st['seconds']:>6.2f}s")

    with open(OUT, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    run()
