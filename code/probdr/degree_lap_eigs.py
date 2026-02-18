
import numpy as np
import matplotlib.pyplot as plt; plt.ion()

n = 500

fig, axs = plt.subplots(1, 4)

def run_expt(A, i):
    A = (A + A.T).clip(0, 1)
    L = np.diag(A.sum(0)) - A

    axs[i].plot(np.sort(np.diag(L)), label='degree', alpha=0.5)
    axs[i].plot(np.linalg.eigvalsh(L), label='eigenvalue', alpha=0.5)

p = 0.001
A = np.random.binomial(1, p, size=(n, n))
run_expt(A, 0)

p = 0.05
A = np.random.binomial(1, p, size=(n, n))
run_expt(A, 1)

A = np.random.multinomial(15, np.ones(n)/n, size=(n, ))
run_expt(A, 2)

ds = np.round(np.abs(np.random.standard_t(1, n)*10))
A = np.concatenate([np.random.multinomial(min(ds[i], n-1), np.ones(n)/n)[None, :] for i in range(len(ds))], axis=0)
run_expt(A, 3)


axs[0].set_ylabel('value')
axs[0].set_xlabel('index')

axs[0].set_title('A_ij ~ Bern(0.001)')
axs[1].set_title('A_ij ~ Bern(0.05)')
axs[2].set_title('A_i  ~ MHypGeom(15, 1/n)')
axs[3].set_title('L_ii ~ HalfCauchy(10)')

plt.tight_layout()