import numpy as np
from core import grad, loss

def sgd(X, y, lr, batch, steps, seed=0):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    for t in range(steps):
        idx = rng.integers(0, n, batch) #random minibatch
        w = w - lr * grad(w, X[idx], y[idx])
    return w

def gd(X, y, lr, steps): #full batch, batch = n
    d = X.shape[1]
    w = np.zeros(d)
    for t in range(steps):
        w = w - lr * grad(w, X, y)
    return w

def hvp(v, X): #return Hv
    n = X.shape[0]
    return X.T @ (X @ v) / n

#use cg(lambda v: hvp(v, X), -g, max_iter)
def cg(matvec, b, tol=1e-10, max_iter=None): #Solve x in Ax = b, A given only as matvec(v) -> A@v
    d = b.shape[0] #A is (d,d), so d comes from b now, not A.shape
    if max_iter is None:
        max_iter=d
    x = np.zeros(d)
    r = b.copy() #since r = b-Ax = b because x = 0
    p = r
    while (np.linalg.norm(r) > tol) and max_iter != 0:
        Ap = matvec(p)
        alpha = ((r.T)@r)/(p.T @ Ap)
        x = x + alpha * p
        r_curr = r
        r = b-matvec(x)
        coef = (r.T @ r)/(r_curr.T @ r_curr)
        p = r + coef * p
        max_iter -= 1
    return x

def hf(X, y, steps=1, tol=1e-10, max_iter=None): #exact Newton, no lr, no damping
    d = X.shape[1]
    w = np.zeros(d)
    for t in range(steps):
        g = grad(w, X, y)
        p = cg(lambda v: hvp(v, X), -g, tol, max_iter) #solve Hp = -g, H never formed
        w = w + p
    return w