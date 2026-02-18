
import gc, torch
import numpy as np
from tqdm import trange
from pykeops.torch import LazyTensor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt; plt.ion()

np.random.seed(42); torch.manual_seed(42)

x, y = torch.load('mnist.pth', weights_only=False)
x = x[:20000]
y = y[:20000]
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
eps = n_bar / n

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
H = I - O/n
L = A.sum(0).diag() - A

_l, _U = torch.linalg.eigh(L/n)
init = _U[:, [1, 2]] @ _l[[1, 2]].reciprocal().sqrt().diag()

def bound_bern():
    X = model.weight #- model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    p = 1/(1 + D*scale)
    return -torch.distributions.Bernoulli(eps*p.clip(1e-4, 1-1e-4)).log_prob(A).sum()

scales = [0.05, 0.1, 0.5, 1, 5, 10]
fig, axs = plt.subplots(1, len(scales))

# init = PCA(n_components=2).fit_transform(x)
init /= (init[:, 0].std())
init = torch.tensor(init).float().cuda()

for i in range(len(scales)):
    scale = scales[i]
    model = torch.nn.Embedding.from_pretrained(init.clone()/scale, freeze=False).to("cuda")

    torch.manual_seed(0)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0, lr=.1)
    for epoch in (bar := trange(100, leave=False)):
        optimizer.param_groups[0]["lr"] = 0.1*(1 - epoch/100)
        model.train()
        loss = bound_bern()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.set_description(f"L:{loss.item()}")
    
    gc.collect(); torch.cuda.empty_cache()

    embd = model.weight.detach().cpu().numpy()

    axs[i].scatter(*embd.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
    axs[i].set_title(f"scale:{np.round(scale, 2)}")
    axs[i].axis('off')

plt.tight_layout()