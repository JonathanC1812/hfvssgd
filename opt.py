import numpy as np
from core import grad, loss

def _stats(n_grad=0.0, n_hvp=0.0, steps=0, converged=False):
    return {"n_grad": n_grad, "n_hvp": n_hvp, "cost": n_grad + n_hvp,
            "steps": steps, "converged": converged}

def sgd(X, y, lr, batch=10, tol=1e-6, L_star=0.0, max_steps=10**7, check_every=1, seed=0, mu=None):
    '''
    mu=None -> lr konstan. Bener di interpolation regime (y noiseless): tiap minibatch
    gradient persis 0 di w*, jadi variansnya ilang di optimum dan lr konstan converge linear.
    mu=lambda_min -> jadwal 1/t standar buat strongly convex, lr_t = lr/(1 + lr*mu*t).
    Wajib begitu kalau ada noise: dengan lr konstan SGD mentok di lantai O(lr*sigma^2/mu)
    dan threshold di bawah lantai itu gak akan pernah kesampaian.
    '''
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    target = L_star + tol * (loss(w, X, y) - L_star)
    per_step = batch / n #minibatch lebih murah dari full grad, harus keitung
    for t in range(max_steps):
        if t % check_every == 0 and loss(w, X, y) <= target:
            return w, _stats(n_grad=t * per_step, steps=t, converged=True)
        idx = rng.integers(0, n, batch) #random minibatch
        lr_t = lr if mu is None else lr / (1.0 + lr * mu * t)
        w = w - lr_t * grad(w, X[idx], y[idx])
    return w, _stats(n_grad=max_steps * per_step, steps=max_steps, converged=False)

def gd(X, y, lr, tol=1e-6, L_star=0.0, max_steps=10**7, check_every=1): #full batch, batch = n
    d = X.shape[1]
    w = np.zeros(d)
    target = L_star + tol * (loss(w, X, y) - L_star)
    for t in range(max_steps):
        if t % check_every == 0 and loss(w, X, y) <= target:
            return w, _stats(n_grad=t, steps=t, converged=True)
        w = w - lr * grad(w, X, y)
    return w, _stats(n_grad=max_steps, steps=max_steps, converged=False)

def hvp(v, X): #return Hv
    n = X.shape[0]
    return X.T @ (X @ v) / n

#use cg(lambda v: hvp(v, X), -g, max_iter)
def cg(matvec, b, tol=1e-10, max_iter=None): #Solve x in Ax = b, A given only as matvec(v) -> A@v
    d = b.shape[0] #A is (d,d), so d comes from b now, not A.shape
    if max_iter is None:
        max_iter=10*d
    x = np.zeros(d)
    r = b.copy() #since r = b-Ax = b because x = 0
    p = r
    tol = tol * np.linalg.norm(b)
    while (np.linalg.norm(r) >= tol) and max_iter != 0:
        Ap = matvec(p)
        alpha = ((r.T)@r)/(p.T @ Ap)
        x = x + alpha * p
        rr_old = (r.T) @ r
        r = r - alpha * Ap #reuse Ap, jangan matvec kedua kali
        coef = (r.T @ r)/rr_old
        p = r + coef * p
        max_iter -= 1
    return x

def hf(X, y, tol=1e-6, L_star=0.0, max_outer=200, cg_tol=1e-6, cg_max_iter=None):
    d = X.shape[1]
    w = np.zeros(d)
    target = L_star + tol * (loss(w, X, y) - L_star)
    n_hvp = [0]
    def matvec(v):
        n_hvp[0] += 1
        return hvp(v, X)
    for t in range(max_outer):
        if loss(w, X, y) <= target:
            return w, _stats(n_grad=t, n_hvp=n_hvp[0], steps=t, converged=True)
        g = grad(w, X, y)
        p = cg(matvec, -g, cg_tol, cg_max_iter) #solve for p Hp = -g
        w = w + p
    return w, _stats(n_grad=max_outer, n_hvp=n_hvp[0], steps=max_outer, converged=False)