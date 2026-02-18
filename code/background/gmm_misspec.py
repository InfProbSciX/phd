
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt; plt.ion()

np.random.seed(42)

def make_t_mixture(n=10000, d=2, df=3):
    k = 2
    z = np.random.choice(k, size=n, p=np.ones(k) / k)
    X = np.empty((n, d))
    for j in range(k):
        idx = np.where(z == j)[0]
        if idx.size == 0: continue
        T = np.random.standard_t(df=df, size=(idx.size, d))
        X[idx] = (-1 if j == 0 else 1) * 2.5 + T

    X = StandardScaler().fit_transform(X)
    return X

def posterior_over_k(X, Ks, r=0.1, n_init=5, max_iter=10000):
    """Approx posterior P(k|X) ∝ exp(-BIC_k/2) * Geom(r)."""
    bics = []
    for k in Ks:
        gm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=1e-6,
            random_state=0,
        )
        gm.fit(X)
        bics.append(gm.bic(X))
    bics = np.array(bics)
    log_marg = -0.5 * bics
    prior = r * (1 - r) ** (np.array(Ks) - 1)
    log_post = log_marg + np.log(prior)
    log_post -= log_post.max()
    post = np.exp(log_post)
    return post / post.sum()

Ns = [100, 500, 1000, 3000]
Ks = np.arange(2, 9)
X_full = make_t_mixture()

plt.figure(figsize=(9, 4.2))
for N in Ns:
    X = X_full[:N]
    pk = posterior_over_k(X, Ks, r=0.1, n_init=10, max_iter=500)
    plt.plot(Ks, pk, label=f"N={N}")

plt.axvline(2, ls="--", color="k", alpha=0.7, label="ground truth k=2")
plt.xlabel("k")
plt.ylabel("P(k | data)")
plt.title("Gaussian mixture fit to 2-cluster Student-t data (misspecified)")
plt.legend()
plt.tight_layout()
