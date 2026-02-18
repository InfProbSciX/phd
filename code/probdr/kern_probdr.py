
import gc, torch
import numpy as np
from tqdm import trange
from pykeops.torch import LazyTensor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt; plt.ion()

np.random.seed(42); torch.manual_seed(42)

x, y = torch.load('mnist.pth', weights_only=False)
x = x[:10000]
y = y[:10000]
n = len(x)

k = 15

x_cuda = torch.tensor(x).to("cuda").contiguous()
dists = (LazyTensor(x_cuda[:, None]) - LazyTensor(x_cuda[None])).square().sum(-1)
knn_idx = dists.argKmin(K=k + 1, dim=0)[:, 1:].cpu()
del x_cuda, dists

r, c = torch.triu_indices(n, n, 1)

nn = torch.zeros(n, n).to("cuda")
nn[torch.arange(n)[:, None], knn_idx] = 1
A = (nn + nn.T).clip(0, 1)

n_bar = 2*15*5/1.5
eps = n_bar / (2*n)

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
H = I - O/n
L = A.sum(0).diag() - A

_l, _U = torch.linalg.eigh(L/n)
init = _U[:, [1, 2]] @ _l[[1, 2]].reciprocal().sqrt().diag()
model = torch.nn.Embedding.from_pretrained(init.clone(), freeze=False).to("cuda")

# m = 128
# W = (torch.randn(2, m, device=L.device))
# b = (2 * np.pi) * torch.rand(m, device=L.device)
# R = torch.tensor(special_ortho_group(len(Phi.T)).rvs()).float().cuda()

def bound_cne():
    X = model.weight - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    return (A*D.log1p()).sum() - eps*((1 - A) * (D/(1 + D)).clip(1e-6).log()).sum()

def bound_bern():
    X = model.weight - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    p = 1/(1 + D)
    return -torch.distributions.Bernoulli(eps*p.clip(1e-6, 1 - 1e-6)).log_prob(A).sum()

def bound_pois():
    X = model.weight - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    p = 1/(1 + D)
    return -torch.distributions.Poisson(eps*p.clip(1e-6)).log_prob(A).sum()

def bound_interp():
    X = model.weight - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    # Phi = torch.cat([X, torch.cos(X @ W + b) * np.sqrt(1/m)], axis=1) @ R
    p = 1/(1 + D)
    # S = I/(2*eps) + Phi @ Phi.T
    S = I/(2*eps) + 0.5*(H @ p @ H) + X@X.T # Phi @ Phi.T
    return (L @ S).trace() - n*S.logdet()
    # return (Lp @ torch.linalg.inv(S/n**2)).trace() + n*S.logdet()
    # return (A - eps*p).square().sum()

def bound_interp_wo_cen():
    X = model.weight - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    p = 1/(1 + D)
    S = I/(2*eps) + 0.5*p + X@X.T
    return (L @ S).trace() - n*S.logdet()

torch.manual_seed(0)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=1.)
for epoch in (bar := trange(100, leave=False)):
    optimizer.param_groups[0]["lr"] = 1 - epoch/100
    model.train()
    loss = bound_interp_wo_cen()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    bar.set_description(f"L:{loss.item()}")
    # gc.collect(); torch.cuda.empty_cache()

embd = model.weight.detach().cpu().numpy()

plt.scatter(*embd.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()

fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
axs[0, 0].scatter(embd[:, 0], embd[:, 1], _U[:, 1].cpu(), s=1, cmap="tab10", c=y)
axs[0, 1].scatter(embd[:, 0], embd[:, 1], _U[:, 2].cpu(), s=1, cmap="tab10", c=y)
axs[0, 0].set_xlabel("sol_0"); axs[0, 0].set_ylabel("sol_1"); axs[0, 0].set_zlabel("eig(L/n)_0")
axs[0, 1].set_xlabel("sol_0"); axs[0, 1].set_ylabel("sol_1"); axs[0, 1].set_zlabel("eig(L/n)_1")

axs[1, 0].scatter(_U[:, 1].cpu(), _U[:, 2].cpu(), embd[:, 0], s=1, cmap="tab10", c=y)
axs[1, 1].scatter(_U[:, 1].cpu(), _U[:, 2].cpu(), embd[:, 1], s=1, cmap="tab10", c=y)
axs[1, 0].set_xlabel("eig(L/n)_0"); axs[1, 0].set_ylabel("eig(L/n)_1"); axs[1, 0].set_zlabel("sol_0")
axs[1, 1].set_xlabel("eig(L/n)_0"); axs[1, 1].set_ylabel("eig(L/n)_1"); axs[1, 1].set_zlabel("sol_1")

# le

plt.scatter(*init.cpu().T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()

# pca & gplvm

x_std = torch.tensor(x/255).to("cuda")

init = torch.tensor(PCA(2).fit_transform(x_std.cpu())).to("cuda")

plt.scatter(*init.cpu().T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
mu = x_std.mean(axis=0)[None, :]
model = torch.nn.Embedding.from_pretrained(torch.tensor(init), freeze=False).to("cuda")
log_p = torch.nn.Parameter(torch.ones(4).to("cuda"))
torch.manual_seed(0)
optimizer = torch.optim.Adam(list(model.parameters()) + [log_p], weight_decay=0.0, lr=0.05)
for epoch in (bar := trange(150, leave=False)):
    model.train()

    X = model.weight - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)

    p_o, p_l, p_r, p_n = log_p.sigmoid()

    if epoch <= 10:
        X = X.detach()
        D = D.detach()

    loss = -torch.distributions.MultivariateNormal(O[[0], :], p_l*(X @ X.T) + p_r/(1 + D) + p_o*O + p_n*I).log_prob(x_std.T).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    bar.set_description(f"L:{loss.item()}")
    gc.collect(); torch.cuda.empty_cache()

embd = model.weight.detach().cpu().numpy()

plt.scatter(*embd.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()
