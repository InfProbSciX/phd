
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

torch.manual_seed(42)

nc = 5000

mu_a = torch.tensor([-2.0, 0.0])
mu_b = torch.tensor([ 2.0, 1.0])

Sigma = torch.tensor([[1.0, 0.6],
                      [0.6, 1.5]])

dist_a = torch.distributions.MultivariateNormal(mu_a, covariance_matrix=Sigma)
dist_b = torch.distributions.MultivariateNormal(mu_b, covariance_matrix=Sigma)

X_a = dist_a.sample((nc,))
X_b = dist_b.sample((nc,))

X = torch.cat([X_a, X_b], dim=0)
y = torch.cat([torch.zeros(nc), torch.ones(nc)]).long()

model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=1000,
)
model.fit(X, y)

log_pa = model.predict_log_proba(X)[:nc, 0]
lp_a = dist_a.log_prob(X_a)

df = pd.DataFrame(dict(log_c=log_pa, lp=np.round(lp_a, 1), n=1))
df = df.groupby('lp').aggregate(dict(log_c='mean', n='sum')).reset_index()
df = df.loc[df.n > 100]
plt.scatter(df.lp, df.log_c, s=1)
