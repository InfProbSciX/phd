
import numpy as np
from tqdm import trange
from scipy.stats import norm
import matplotlib.pyplot as plt; plt.ion()

np.random.seed(42)

n, m = 5000, 64
v, k = 1, 1

# def apQ():
#     w = (m * k) / ((m * v**2) + (n * k * v))
#     Q = (1.0 / v) * np.eye(n) - w*(Phi @ Phi.T)
#     return Q

def apQ():
    sigma_x2 = 9.0

    G = W @ W.T
    s = np.diag(G)
    Mdiff = s[:, None] + s[None, :] - 2.0 * G
    Msum  = s[:, None] + s[None, :] + 2.0 * G
    Cdiff = np.cos(b[:, None] - b[None, :])
    Csum  = np.cos(b[:, None] + b[None, :])

    S_hat = (n / m) * (Cdiff * np.exp(-0.5 * sigma_x2 * Mdiff)
                     + Csum  * np.exp(-0.5 * sigma_x2 * Msum))

    D_hat = np.linalg.inv(np.eye(m) + (k / v) * S_hat.astype(np.float64))

    Q = (1.0 / v) * np.eye(n) - (k / (v**4)) * (Phi @ D_hat @ Phi.T)
    return Q

X = np.random.normal(0.0, 3.0, size=(n, 1))
X = X[np.argsort(X[:, 0]), :]

W = np.random.standard_normal((m, 1))
b = np.random.uniform(0, 2*np.pi, size=(m,))
Phi = np.sqrt(2/m) * np.cos(X@W.T + b)

K = k*Phi @ Phi.T + v*np.eye(n)

Q_ap = apQ()
Kap = np.linalg.inv(Q_ap.astype(np.float64))

plt.plot(np.linalg.inv(K.astype(np.float64))[n//2, (n//2 + 1):])
plt.plot(Q_ap[n//2, (n//2 + 1):])

plt.plot(np.linalg.inv(K)[0, 1:])
plt.plot(Q_ap[0, 1:])

plt.plot(Kap[n//2, (n//2 + 1):])
plt.plot(K[n//2, (n//2 + 1):])
