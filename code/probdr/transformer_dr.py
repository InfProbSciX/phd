import torch
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

torch.set_grad_enabled(False)
np.random.seed(42); torch.manual_seed(42)

############################################################################
# Hypers

n, q, lr = 10000, 128, 0.4
num_blocks = 8

############################################################################
# Transformer block inits

block = torch.nn.TransformerEncoderLayer(
    d_model=q,
    nhead=1,
    dim_feedforward=q,
    dropout=0.0,
    activation=torch.nn.Identity(),
    norm_first=False,
)

for k, p in dict(block.named_parameters()).items():
    p.requires_grad_(False)
    if 'bias' in k: p.zero_()
    if 'weight' in k:
        if 'linear' in k: p[:, :] = torch.eye(len(p))
        elif 'self_attn.in_proj_weight' in k:
            p[:, :] = torch.cat([
                torch.eye(q) * np.sqrt(30*n),
                torch.eye(q) * np.sqrt(30*n/q),
                torch.eye(q) * 2 * lr,
            ], axis=0)
        elif 'self_attn.out_proj.weight' in k:
             p[:, :] = torch.eye(q)
        elif k == 'norm1.weight': p[:] = 1/np.sqrt(n)
        elif k == 'norm2.weight': p[:] = 1/np.sqrt(n)
        else: p[:] = 1
    if k == 'linear2.weight':
        p[:, :] = -2 * lr * torch.eye(q)

block.norm1.eps = 1e-10
block.norm2.eps = 1e-10

block = block.cuda()

mask = torch.zeros(n, n).cuda()
mask.fill_diagonal_(float('-inf'))

############################################################################
# Data inits

x, y = torch.load('mnist.pth', weights_only=False)
x = x[:n]
y = y[:n]

# init = PCA(n_components=q).fit_transform(x)
init = x @ np.random.normal(size=(len(x.T), q))/np.sqrt(len(x.T))
X = torch.tensor(init).float().cuda()

############################################################################
# Transformer

X = block.norm2(X)
for epoch in range(num_blocks):
    X = block(X, src_mask=mask)

############################################################################
# Plots

X_init_2 = init[:, :2]
X_final = X.cpu().numpy()[:, :2]

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(X_init_2[:, 0], X_init_2[:, 1], c=y, alpha=0.75, s=1, cmap="tab10", edgecolors="none")
axs[1].scatter(X_final[:, 0], X_final[:, 1], c=y, alpha=0.75, s=1, cmap="tab10", edgecolors="none")

for ax in axs: ax.set_axis_off()
plt.tight_layout()
