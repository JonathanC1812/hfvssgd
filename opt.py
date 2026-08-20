import numpy as np
from core import grad, loss

def _stats(n_grad=0.0, n_hvp=0.0, steps=0, reached_floor=False, diverged=False, trace=None):
    return {"n_grad": n_grad, "n_hvp": n_hvp, "cost": n_grad + n_hvp,
            "steps": steps, "reached_floor": reached_floor, "diverged": diverged,
            "trace": np.asarray(trace, dtype=float).reshape(-1, 2)}


def _floor(L0, L_star, eps_floor):
    return L_star + eps_floor * (L0 - L_star)

def sgd(X, y, lr=None, batch=10, budget=1e5, eps_floor=0.0, L_star=0.0,
        check_every=None, seed=0, lmin=None, lmax=None,
        lr_grid=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5), tune_budget=2_000, eps_tune=1e-4):
    n, d = X.shape
    if lr is None:
        best = None
        for c in lr_grid:
            cand = c / lmax
            w_p, st_p = sgd(X, y, cand, batch=batch, budget=tune_budget,
                            eps_floor=eps_tune, L_star=L_star,
                            check_every=check_every, seed=seed, lmin=lmin)
            if st_p["diverged"]:
                key = (True, np.inf)
            else:
                tr = st_p["trace"]
                c_eps = cost_to_eps(tr, tr[0, 1], L_star, eps_tune)
                key = (c_eps is None, c_eps if c_eps is not None else loss(w_p, X, y))
            if best is None or key < best[0]:
                best = (key, cand)
        lr = best[1]
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    L0 = loss(w, X, y)
    floor = _floor(L0, L_star, eps_floor)
    per_step = batch / n
    if check_every is None:
        check_every = max(1, int(round(n / batch)))
    max_steps = int(budget / per_step)

    trace = [(0.0, L0)]
    reached = L0 <= floor
    diverged = False
    t = 0
    while t < max_steps and not reached:
        idx = rng.integers(0, n, batch)
        lr_t = lr if lmin is None else lr / (1.0 + lr * lmin * t)
        w = w - lr_t * grad(w, X[idx], y[idx])
        t += 1
        if not np.isfinite(w).all():
            diverged = True
            break
        if t % check_every == 0:
            L = loss(w, X, y)
            trace.append((t * per_step, L))
            reached = L <= floor
    if diverged:
        trace.append((t * per_step, np.inf))
    elif t % check_every != 0:
        trace.append((t * per_step, loss(w, X, y)))
    st = _stats(n_grad=t * per_step, steps=t, reached_floor=reached,
                diverged=diverged, trace=trace)
    st.update(lr=lr, batch=batch)
    return w, st


def gd(X, y, lr, budget=1e5, eps_floor=0.0, L_star=0.0, check_every=1):
    d = X.shape[1]
    w = np.zeros(d)
    L0 = loss(w, X, y)
    floor = _floor(L0, L_star, eps_floor)
    max_steps = int(budget)
    trace = [(0.0, L0)]
    reached = L0 <= floor
    t = 0
    while t < max_steps and not reached:
        w = w - lr * grad(w, X, y)
        t += 1
        if t % check_every == 0:
            L = loss(w, X, y)
            trace.append((float(t), L))
            reached = L <= floor
    if t % check_every != 0:
        trace.append((float(t), loss(w, X, y)))
    return w, _stats(n_grad=float(t), steps=t, reached_floor=reached, trace=trace)


def hvp(v, X):  # return Hv
    n = X.shape[0]
    return X.T @ (X @ v) / n


# use cg(lambda v: hvp(v, X), -g, max_iter)
def cg(matvec, b, tol=1e-10, max_iter=None, on_iter=None):  # Solve x in Ax = b, A given only as matvec(v) -> A@v
    d = b.shape[0]  # A is (d,d), so d comes from b now, not A.shape
    if max_iter is None:
        max_iter = 10 * d
    x = np.zeros(d)
    r = b.copy()  # since r = b-Ax = b because x = 0
    p = r
    tol = tol * np.linalg.norm(b)
    while (np.linalg.norm(r) >= tol) and max_iter != 0:
        Ap = matvec(p)
        alpha = ((r.T) @ r) / (p.T @ Ap)
        x = x + alpha * p
        rr_old = (r.T) @ r
        r = r - alpha * Ap
        coef = (r.T @ r) / rr_old
        p = r + coef * p
        max_iter -= 1
        if on_iter is not None:
            on_iter(x)
    return x


def hf(X, y, budget=1e5, eps_floor=0.0, L_star=0.0, cg_tol=1e-6,
       cg_max_iter=None, max_outer=200):
    d = X.shape[1]
    w = np.zeros(d)
    L0 = loss(w, X, y)
    floor = _floor(L0, L_star, eps_floor)

    n_hvp = [0]
    def matvec(v):
        n_hvp[0] += 1
        return hvp(v, X)

    trace = [(0.0, L0)]
    reached = L0 <= floor
    t = 0
    while t < max_outer and not reached and (t + n_hvp[0]) < budget:
        g = grad(w, X, y)
        t += 1
        w_outer = w 
        def on_iter(x, _t=t, _w=w_outer):
            trace.append((_t + n_hvp[0], loss(_w + x, X, y)))
        p = cg(matvec, -g, cg_tol, cg_max_iter, on_iter=on_iter)  # solve for p in Hp = -g
        w = w + p
        L = loss(w, X, y)
        trace.append((t + n_hvp[0], L))
        reached = L <= floor
    return w, _stats(n_grad=float(t), n_hvp=float(n_hvp[0]), steps=t,
                     reached_floor=reached, trace=trace)

def cost_to_eps(trace, L0, L_star, eps):
    target = _floor(L0, L_star, eps)
    hit = trace[trace[:, 1] <= target]
    return float(hit[0, 0]) if len(hit) else None


def acc_at_budget(trace, L0, L_star, C):
    seen = trace[trace[:, 0] <= C]
    if not len(seen):
        return 1.0
    return float((seen[-1, 1] - L_star) / (L0 - L_star))
