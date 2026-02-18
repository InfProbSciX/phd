import argparse, os, gc
import esm, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

np.random.seed(42); torch.manual_seed(42)

MODEL_NAMES = [f"esm2_{sb}_UR50D" for sb in [
    "t6_8M",
    "t12_35M",
    "t30_150M",
    "t33_650M",
    "t36_3B",
    "t48_15B",
]]

# # download proteingym DMS benchmarks to curdir
# DFs = []
# for file in tqdm(os.listdir(), leave=False):
#     if not file.endswith(".csv"):
#         continue

#     df = pd.read_csv(os.path.join(file))
#     seqs = df["mutated_sequence"].tolist()

#     if (len(seqs[0]) > 500) or (len(seqs) < 30):
#         continue

#     df = pd.read_csv(os.path.join(file))
#     np.random.seed(42)
#     df = df.iloc[
#         np.random.choice(len(df), min(200, len(df)), replace=False)
#     ].reset_index(drop=True)

#     DFs.append(df)
# df_idxs = np.random.choice(len(DFs), 50, replace=False)
# DFs = [df for (i, df) in enumerate(DFs) if i in df_idxs]
# torch.save(DFs, 'DFs.pkl')

DFs = torch.load('DFs.pkl', weights_only=False)

def score_batch(seqs):
    batch = [(str(i), seq) for i, seq in enumerate(seqs)]
    labels, seqs, tokens = batch_converter(batch)
    tokens = tokens.cuda() if next(model.parameters()).is_cuda else tokens

    with torch.no_grad(), torch.inference_mode(), torch.amp.autocast('cuda'):
        lps = []
        for _ in range(1 if dropout == 0 else 100):
            lp_i = model(tokens, repr_layers=[], return_contacts=False)["logits"]
            lp_i = torch.log_softmax(lp_i, dim=-1)
            lps.append(
                lp_i[None, ...]
            )
            gc.collect(); torch.cuda.empty_cache()

        lp = torch.cat(lps, axis=0).mean(0)
        lp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        lp = lp.sum(dim=1).cpu().numpy()

    return lp

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--model_idx", type=int, default=1)
    # args.add_argument("--dropout", type=float, default=0.1)
    args = args.parse_args()

    # dropout = args.dropout
    model_idx = args.model_idx
    print(model_idx)

    for dropout in [0.0, 0.1, 0.2, 0.3, 0.4]:
        np.random.seed(42); torch.manual_seed(42)

        model, alphabet = esm.pretrained.load_model_and_alphabet(
            MODEL_NAMES[model_idx]
        )
        batch_converter = alphabet.get_batch_converter()
        model.train()

        # dropouts
        model.embed_tokens = torch.nn.Sequential(
            model.embed_tokens,
            torch.nn.Dropout(dropout),
        )

        model = model.half()

        for (i, layer) in enumerate(model.layers):
            if (i <= len(model.layers)//(3 if model_idx == 5 else 5)) and (model_idx in [2, 5]):
                layer.self_attn.out_proj = torch.nn.Sequential(
                    layer.self_attn.out_proj,
                    torch.nn.Dropout(dropout)
                )

        ####################################

        if torch.cuda.is_available():
            model = model.cuda()

        all_rhos = []
        for df in (bar := tqdm(DFs, leave=False)):
            seqs = df["mutated_sequence"].tolist()
            dms_scores = df["DMS_score"].values

            if model_idx < len(MODEL_NAMES) - 1:
                scores = score_batch(seqs)
            else:
                gc.collect(); torch.cuda.empty_cache()
                batch_size = 25; scores = []
                for i in range(0, len(seqs), batch_size):
                    batch = seqs[i : i + batch_size]
                    scores.extend(score_batch(batch))

            rho, _ = spearmanr(scores, dms_scores)
            all_rhos.append(np.abs(rho))
            bar.set_description(f"n:{len(all_rhos)},SRCC:{np.round(np.median(all_rhos)*100, 1)}%")

        with open(f"results_final/etr_id_{model_idx},dp_{dropout},p_{np.round(np.median(all_rhos)*100, 1)}", 'w') as file:
            file.write("".join([f"{x}," for x in all_rhos]))
