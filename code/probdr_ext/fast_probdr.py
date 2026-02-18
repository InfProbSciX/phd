
## SGNS

import gc, torch
import numpy as np
from tqdm import trange, tqdm
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

n_bar = 2*k*5/1.5
eps = n_bar / n

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
H = I - O/n
L = A.sum(0).diag() - A

Di = A.sum(0).sqrt().reciprocal().diag()
At = Di @ A @ Di
At = torch.linalg.matrix_power(At, 10) * k * 1.5

K = -0.5 * H @ (2*eps/At).log1p() @ H

l, U = torch.linalg.eigh(K)
embd = U[:, [-1, -2]].cpu().numpy()

plt.scatter(*embd[:, :2].T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()

## Parametric

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
eps = n_bar / (2*n)

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
H = I - O/n
L = A.sum(0).diag() - A

_l, _U = torch.linalg.eigh(L/n)
num_comps = 8
init = _U[:, 1:(num_comps + 1)] @ _l[1:(num_comps + 1)].reciprocal().sqrt().diag()

class Model(torch.nn.Module):
    def __init__(self, init):
        super().__init__()
        
        self.X0 = init.clone()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(num_comps, 2, bias=False),
        )
        self.to(self.X0.device)

    def forward(self):
        return self.nn(self.X0)

model = Model(init)
model.nn[0].weight.data = torch.eye(2, num_comps).cuda() + 1e-2

def bound_interp():
    X = model()
    X = X - X.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    p = 1/(1 + D)
    S = I/(2*eps) + 0.5*(H @ p @ H) + X@X.T
    return (L @ S).trace() - n*S.logdet()

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=1.)
for epoch in range(150):
    model.train()
    loss = bound_interp()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"L:{loss.item()}")

embd = model().detach().cpu().numpy()

plt.scatter(*embd.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()
