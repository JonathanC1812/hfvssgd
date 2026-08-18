import numpy as np

# ---------- data ----------

def whiten(Z): #making k = 1
    n = Z.shape[0]
    U, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return np.sqrt(n) * (U @ Vt)

def spectrum(d, kappa):
    return np.logspace(0, 0.5*np.log10(kappa), d)

def make_data(n, d, kappa, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d))
    Z = whiten(Z)
    s = spectrum(d, kappa)
    X = Z * s
    w_true = rng.standard_normal(d)
    y = X @ w_true
    return X, y, w_true
'''
Z = random n x d matrix
whiten(Z) is to make the variance-covariance matrix (calculated by Z.T @ Z / n = I)
s = spectrum of d amount of number from 1 to sqrt(kappa)
X = dataset where when we want to find the hessian, the kappa will be k = kappa (n x d)
w_true is a random vector of length d just to make y (dx1)
y = answer key for X (nx1)
'''

def calculate_kappa(X):
    #H is from the 2nd derivative of loss of MSE
    n = X.shape[0]
    H = (X.T @ X) / n #matrix with k = kappa (dxd)
    eigen = np.linalg.eigvalsh(H) #symmetric H -> real eigenvalues, ascending
    return eigen[-1] / eigen[0] #k

# ---------- problem ----------

def loss(w, X, y):
    r = X @ w - y
    return (r @ r/(2*X.shape[0])) #1/2n * sum((Xw - y)^2)

def grad(w, X, y): #(d,)
    n = X.shape[0]
    r = X @ w - y
    return X.T @ r / n #dL/dw = X.T (Xw-y) / n = He

if __name__ == "__main__":
    next