
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
np.random.seed(42)

def e_log_f(r, mu_c, d, n_samples=50000, seed=0):
    z = np.random.normal(size=(n_samples, d))
    u = z / np.linalg.norm(z, axis=1, keepdims=True)

    logits = r * u[:, 0] * mu_c + (mu_c**2) / 2.0
    vals = -np.log1p(np.exp(-logits))
    return vals.mean()

mu_cs = [1.0, 3.0]
ds = [3, 8]
r_values = np.linspace(0.0, 4.0, 40)

fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

for i, mu_c in enumerate(mu_cs):
    for j, d in enumerate(ds):
        ax = axes[i, j]
        log_post_means = [e_log_f(r, mu_c, d, seed=42)
                          for r in r_values]
        ax.plot(r_values, log_post_means, marker='o', c='orange')
        ax.set_title(f"mu_c = {mu_c}, d = {d}")
        ax.grid(True)

for ax in axes[-1, :]:
    ax.set_xlabel("r")

plt.tight_layout()
