import os, torch, fairseq, random, numpy as np
device = 'cuda'

model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(['wav2vec_large.pt'])
model = model[0].eval().to(device)
model.wav2vec_predictions.infonce = False

torch.manual_seed(42); np.random.seed(42); random.seed(42)
a = torch.nn.Parameter(torch.randn(1, 80000).to(device))

z = model.feature_extractor(a)
c = model.feature_aggregator(z)

def forward(self=model.wav2vec_predictions, c=c, z=z):

    W = self.project_to_steps.weight[:, :, 0, :]
    b = self.project_to_steps.bias[:, None]

    _, _, K = W.shape  # 12
    _, n_d, n_t = c.shape

    transformed_c = torch.cat([(W[..., k].T @ c[0] + b)[None, ..., None] for k in range(K)], axis=-1)

    n_negatives = 10 # self.n_negatives
    torch.randint(low=0, high=n_t, size=(1, n_negatives * n_t))

    with torch.no_grad():
        n_ts = torch.arange(n_t).repeat_interleave(10)

        neg_idxs = torch.randint(
            low=0, high=n_t - 1, size=(1, n_negatives * n_t)
        )
        neg_idxs[neg_idxs >= n_ts] += 1

    negs = z[0, ..., neg_idxs.view(-1)]
    negs = negs.view(n_d, 1, n_negatives, n_t).permute(2, 1, 0, 3)

    negatives = negs

    z = z.unsqueeze(0)
    targets = torch.cat([z, negatives], dim=0)  # Copies x B x C x T

    targets = targets[:, 0, :, :] # 11, 512, 498
    transformed_c = transformed_c[0] # 512, 498, 12

    preds = []; labs = []
    for i in range(K):
        dot = (targets[..., (i + self.offset):] * transformed_c[None, ..., :-(i + self.offset), i]).sum(1)
        lab = torch.zeros_like(dot)
        lab[0, :] = 1.0
        preds.append(dot.flatten())
        labs.append(lab.flatten())

    return torch.cat(preds), torch.cat(labs)

def test():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    a = torch.nn.Parameter(torch.randn(1, 80000).to(device))

    z = model.feature_extractor(a)
    c = model.feature_aggregator(z)
    x1, x2 = model.wav2vec_predictions(c, z)

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    a = torch.nn.Parameter(torch.randn(1, 80000).to(device))

    z = model.feature_extractor(a)
    c = model.feature_aggregator(z)
    y1, y2 = forward(self=model.wav2vec_predictions, c=c, z=z)

    assert (y2 == x2).all()
    assert abs(y1 - x1).max() < 0.005

test()