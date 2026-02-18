import os
import torch
import numpy as np
from tqdm import tqdm
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from model import LayerNorm, MLP, GPTConfig, Block, CausalSelfAttention
import random

# random.seed(42); np.random.seed(42); torch.manual_seed(42)

EYE = False

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FullAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(self, 'bias'):
            del self.bias
        if hasattr(self, 'register_buffer'):
            self.register_buffer('bias', None)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = (att - (0 if (not EYE) else I)) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class ViTBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = FullAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class VisionGPT(torch.nn.Module):
    def __init__(
        self, 
        image_size=64,
        patch_size=16,
        num_classes=1000,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        channels=3
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(3, dim, kernel_size=4, stride=4),
            Rearrange('b c h w -> b (h w) c'),
            LayerNorm(dim, bias=True)
        )

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = torch.nn.Dropout(emb_dropout)

        # Create GPT-style config for blocks
        config = GPTConfig(
            n_embd=dim,
            n_head=heads,
            dropout=dropout,
            bias=True,
            block_size=num_patches+1,  # Not used but required for config
            vocab_size=None             # Not used
        )

        self.blocks = torch.nn.Sequential(*[
            ViTBlock(config) for _ in range(depth)
        ])

        self.ln_f = LayerNorm(dim, bias=True)
        self.head = torch.nn.Linear(dim, num_classes)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, img):
        # Reshape and normalize
        img = img.reshape(-1, 3, 16, 16).float() / 255
        img = (img - self.mean) / self.std

        # if model.training:
        #     flip_mask = torch.rand(img.size(0), device=img.device) < 0.5
        #     img[flip_mask] = torch.flip(img[flip_mask], [3])

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x[:, 0])

if __name__ == '__main__':
    # np.random.seed(42); torch.manual_seed(42)

    data_path = '../imagenet/imagenet.pth'
    X, y, v = torch.load(data_path, weights_only=False)

    train_dataset = torch.utils.data.TensorDataset(
        X[v == 0], y[v == 0]
    )
    val_dataset = torch.utils.data.TensorDataset(
        X[v == 1], y[v == 1]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = VisionGPT(
        image_size=16,
        patch_size=4,
        num_classes=1000,
        dim=256,
        depth=12,
        heads=4,
        dropout=0.1,
        emb_dropout=0.1
    ).cuda()

    I = torch.eye(np.square(16//4) + 1, device="cuda")[None, None, :, :]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(50):
        model.train()
        train_correct = []
        train_losses = []

        for x, c in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, c = x.cuda(), c.cuda() - 1

            logits = model(x)
            loss = F.cross_entropy(logits, c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_correct.append((logits.argmax(1) == c).cpu())
            train_losses.append(loss.item())

        train_acc = torch.cat(train_correct).float().mean().item()
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_correct = []
        val_losses = []
        with torch.no_grad():
            for x, c in val_loader:
                x, c = x.cuda(), c.cuda() - 1
                logits = model(x)
                loss = F.cross_entropy(logits, c)
                val_losses.append(loss.item())
                val_correct.append((logits.argmax(1) == c).cpu())

        val_acc = torch.cat(val_correct).float().mean().item()
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch}: "
              f"Train Acc: {train_acc*100:.1f}%, "
              f"Val Acc: {val_acc*100:.1f}%")
