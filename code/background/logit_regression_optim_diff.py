import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt; plt.ion()

n = 1000
df = 100
cen = 10

np.random.seed(42)

X = torch.tensor(np.concatenate([
    np.random.standard_t(df, size=(n//2, 2)) - np.ones((1, 2))*cen,
    np.random.standard_t(df, size=(n//2, 2)) + np.ones((1, 2))*cen,
])).float()

y = torch.tensor(np.concatenate(
    [np.zeros(n//2), np.ones(n//2)])
).float()

for Opt in [torch.optim.SGD, torch.optim.Adam]:

    rec = []
    for _ in trange(1000):
        wa = torch.nn.Parameter(torch.randn(1, 2))

        opt = Opt([wa], lr=0.01)
        for i in range(1000):
            opt.zero_grad()
            p = (((wa)*X).sum(axis=1)).sigmoid()
            nlp = -torch.distributions.Bernoulli(p).log_prob(y).sum()
            nlp.backward()
            opt.step()

        rec.append(torch.atan2(-wa[0, 0], wa[0, 1]).item())

    plt.hist(rec, label=Opt.__name__, alpha=0.75, density=True, bins=40)
plt.legend()