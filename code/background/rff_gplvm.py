
import gc, torch
import numpy as np
from tqdm import trange
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt; plt.ion()

np.random.seed(42); torch.manual_seed(42)

x, y = torch.load('mnist.pth', weights_only=False)
x = x[:10000]
y = y[:10000]
n = len(x)

k = 15
init = PCA(n_components=2).fit_transform(x)
init /= (init[:, 0].std())

model = torch.nn.Embedding.from_pretrained(torch.tensor(init), freeze=False).to("cuda")

x_cuda = torch.tensor(x).to("cuda").contiguous()
x_std = torch.tensor(x / x.std()).to("cuda")

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
mu = x_std.mean(axis=0)[None, :]
model = torch.nn.Embedding.from_pretrained(torch.tensor(init), freeze=False).to("cuda")
log_p = torch.nn.Parameter(torch.ones(4).to("cuda"))
torch.manual_seed(0)
optimizer = torch.optim.Adam(list(model.parameters()) + [log_p], weight_decay=0.0, lr=0.025)
for epoch in (bar := trange(300, leave=False)):
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

##########################################################################################

import gc, torch
import numpy as np
from tqdm import trange
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt; plt.ion()

np.random.seed(42); torch.manual_seed(42)

x, y = torch.load('mnist.pth', weights_only=False)
x = x[:10000]
y = y[:10000]
n = len(x)

k = 15
init = PCA(n_components=2).fit_transform(x)
init /= (init[:, 0].std())

model = torch.nn.Embedding.from_pretrained(torch.tensor(init), freeze=False).to("cuda")

x_cuda = torch.tensor(x).to("cuda").contiguous()
x_std = torch.tensor(x / x.std()).to("cuda")

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
mu = x_std.mean(axis=0)[None, :]
model = torch.nn.Embedding.from_pretrained(torch.tensor(init), freeze=False).to("cuda")

m_rff = 512
d_x   = 2
d_obs = x_std.shape[1]
n     = x_std.shape[0]

torch.manual_seed(0)
W_rff = torch.randn(m_rff, d_x, device="cuda")     # ~ N(0, I)
b_rff = 2*np.pi*torch.rand(m_rff, device="cuda")   # ~ U(0, 2pi)
rff_scale = (2.0 / m_rff) ** 0.5

p = 1 + d_x + m_rff                                 # Phi columns = [1, X, RFFs]
W_out = torch.nn.Parameter(torch.randn(p, d_obs, device="cuda") / np.sqrt(p))
log_sigma = torch.nn.Parameter(torch.tensor(-2.0, device="cuda"))  # noise ~ softplus(log_sigma)

weight_decay_W = 1e-4

opt = torch.optim.Adam(list(model.parameters()) + [W_out, log_sigma], lr=0.01, weight_decay=0.0)

for epoch in (bar := trange(100000, leave=False)):
    model.train()
    X = model.weight - model.weight.mean(0)

    if epoch <= 10: X_used = X.detach()
    else: X_used = X

    proj = X_used @ W_rff.T + b_rff
    Phi_rff = rff_scale * torch.cos(proj)
    Phi = torch.cat([torch.ones(n, 1, device="cuda"), X_used, Phi_rff], dim=1)

    loss = -torch.distributions.Normal(Phi @ W_out, log_sigma.exp()).log_prob(x_std).sum()

    opt.zero_grad()
    loss.backward()
    opt.step()

    bar.set_description(f"PhiMF L:{loss.item():.3f}")

gc.collect(); torch.cuda.empty_cache()

embd_phi = model.weight.detach().cpu().numpy()
plt.figure()
plt.scatter(*embd_phi.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off'); plt.tight_layout()
