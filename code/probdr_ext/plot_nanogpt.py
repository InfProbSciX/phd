import json
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

with open('nanoGPT/results_summary.json') as f:
    vanilla = json.load(f)['best_losses']

with open('nanoGPT/results_summary_eye.json') as f:
    lap = json.load(f)['best_losses']

df = pd.DataFrame({
    'Validation Loss': vanilla + lap,
    'Method': ['Vanilla'] * len(vanilla) + ['Laplacian Smth'] * len(lap)
})

sns.violinplot(
    x='Method',
    y='Validation Loss',
    data=df,
    palette="flare",
    fill=False,
    inner='box',
    cut=0,
    ax=axs[0]
)

axs[0].set_title('nanoGPT Shakespeare')

with open('nanoGPT/imagenet.json') as f:
    vanilla = json.load(f)['best_val_accs']

with open('nanoGPT/imagenet_eye.json') as f:
    lap = json.load(f)['best_val_accs']

df = pd.DataFrame({
    'Validation Accuracy': vanilla + lap,
    'Method': ['Vanilla'] * len(vanilla) + ['Laplacian Smth'] * len(lap)
})

sns.violinplot(
    x='Method',
    y='Validation Accuracy',
    data=df,
    palette="flare",
    fill=False,
    inner='box',
    cut=0,
    ax=axs[1]
)

axs[1].set_title('ViT Imagenet 16*16')

plt.tight_layout()

##########################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt; plt.ion()

df_van = pd.DataFrame([...])
df_eye = pd.DataFrame([...])

df_eye = df_eye.rename(dict(loss='loss_eye'), axis=1)
df = pd.merge(df_van, df_eye, on='iter')
df['diff'] = df.loss - df.loss_eye


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.lineplot(df, x='iter', y='diff', ax=axs[0])
axs[0].hlines(0, 0, 4750, color='orange', linestyles='--')
axs[0].set_ylabel('Train loss diff. (Vanilla - Laplacian Smth)')
axs[0].set_xlabel('Iteration')
axs[0].set_title('nanoGPT-2')

sns.lineplot(df.loc[df.iter > 2000], x='iter', y='loss', ax=axs[1], label='Vanilla')
sns.lineplot(df.loc[df.iter > 2000], x='iter', y='loss_eye', ax=axs[1], label='Lap. Smth.')

plt.tight_layout()
