import numpy as np
import matplotlib.pyplot as plt; plt.ion()
plt.style.use('seaborn-v0_8-pastel')
np.random.seed(42)

def rbf_kernel(X):
    return np.exp(-0.5 * np.square((X - X.T)))

def periodic_kernel(X):
    return np.exp(-2.0 * (np.sin(np.pi * np.abs(X - X.T)) ** 2))

def linear_kernel(X):
    return X @ X.T

def sample_gp_prior(K, n_samples=5, jitter=1e-8, rng=None):
    n = K.shape[0]
    L = np.linalg.cholesky(K + jitter * np.eye(n))
    z = np.random.standard_normal((n, n_samples))
    return L @ z

X = np.linspace(-3, 3, 300)[:, None]

K_smooth   = rbf_kernel(X)
K_periodic = periodic_kernel(X)
K_linear   = linear_kernel(X)

ys_smooth   = sample_gp_prior(K_smooth,   n_samples=3)
ys_periodic = sample_gp_prior(K_periodic, n_samples=3)
ys_linear   = sample_gp_prior(K_linear,   n_samples=3)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
titles = ["rbf kernel", "periodic kernel", "linear kernel"]
samples = [ys_smooth, ys_periodic, ys_linear]

for ax, title, ys in zip(axes, titles, samples):
    ax.set_title(f"{title}")
    for i in range(ys.shape[1]):
        ax.plot(X[:, 0], ys[:, i], lw=1.5, alpha=0.75)
    ax.set_xlabel("x")
    ax.axis('off')

axes[0].set_ylabel("f(x)")
plt.tight_layout()