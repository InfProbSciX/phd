import torch
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from sklearn.decomposition import PCA
from pykeops.torch import LazyTensor
import scipy.sparse

x, y = torch.load('mnist.pth', weights_only=False)
x = x[:10000]
y = y[:10000]
n, d = x.shape

k = 15
init = PCA(n_components=2).fit_transform(x)
init /= (init[:, 0].std())

model = torch.nn.Embedding.from_pretrained(torch.tensor(init), freeze=False).to("cuda")

x_cuda = torch.tensor(x).clone().to("cuda").contiguous()
dists = ((LazyTensor(x_cuda[:, None]) - LazyTensor(x_cuda[None])) ** 2).sum(-1)
knn_idx = dists.argKmin(K=k + 1, dim=0)[:, 1:].cpu()
del x_cuda, dists

nn = torch.zeros(n, n).to("cuda")
nn[torch.arange(n)[:, None], knn_idx] = 1

r, c = torch.triu_indices(n, n, 1)

A = (nn + nn.T).clip(0, 1)
nn = A[r, c]
n_bar = 2*15*5/1.5
_mult = n_bar / n

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
H = I - O/n
L = A.sum(0).diag() - A

negative_samples = 5

s = 0.0

Z_umap = n ** 2 / 5
Z_tsne = 100 * n
Z_bar = Z_umap if s == 1 else Z_tsne
spec_param = negative_samples * Z_bar / n ** 2

optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

spec_param_early = 1.0

n_epochs = 100
torch.manual_seed(0)
for epoch in range(n_epochs):
    if epoch < n_epochs // 3:
        cur_spec_param = spec_param_early
    else:
        cur_spec_param = spec_param

    optimizer.param_groups[0]["lr"] = 1.0 * (1 - epoch / n_epochs)

    X = model.weight - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    p = 1 / (1 + cur_spec_param * (D + 1))

    loss = -(
        (A * torch.log(p.clamp(1e-4))).sum() + torch.log(1 - p.clamp(1e-4)).mean() * A.sum()*5
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

embd = model.weight.detach().cpu().numpy()

plt.scatter(*embd.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()
