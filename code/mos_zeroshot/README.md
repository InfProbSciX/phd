
# Zero-Shot Speech Model Evaluation Toolkit

This repository accompanies our work, [Uncertainty as a Predictor: Leveraging Self-Supervised Learning for Zero-Shot MOS Prediction](https://arxiv.org/abs/2312.15616).

## Overview
This toolkit provides a comprehensive suite of experiments for evaluating various zero-shot speech models. It includes tests on noise, reconstructed audio, and Mean Opinion Score (MOS) prediction on different datasets. The toolkit supports a range of models including Wav2Vec, multilingual, and Chinese Wav2Vec2 models.

## Environment Setup

```bash
login_private.sh
conda create -p ./zeroshot python=3.10 -y
conda activate ./zeroshot
pip install librosa lightning tqdm matplotlib fairseq ipython numpy scipy pandas scipy wandb torch torchaudio pypi-kenlm flashlight-text transformers --extra-index-url https://download.pytorch.org/whl/cu118
```

## Data Structure
The data should be organized as follows, with DATA_PARENT_DIR as the root (we use the ProbMOS folder as the data is exactly the same for both projects).

```
DATA_PARENT_DIR/
│
├── data/
│   ├── ood/
│   ├── main/
│   ├── track1/
│   ├── track2/
│   └── track3/
└── fairseq/                # Directory for fairseq models
```

Note that some fairseq models don't work out of the box. This monkeypatch is required to get them to work properly:
```python
import torch
from omegaconf import DictConfig, open_dict
cp_path = 'fairseq/espeak_en_26lang_m10.pt'
cp = torch.load(cp_path)
wrong_key = ['eval_wer','eval_wer_config', 'eval_wer_tokenizer', 'eval_wer_post_process', 'autoregressive']
cfg = DictConfig(cp['cfg'])
with open_dict(cfg):
    for k in wrong_key:
        cfg.task.pop(k)
cp['cfg'] = cfg
torch.save(cp, 'fairseq/espeak_en_26lang_m10.pt')
```

## Usage

At the moment, the script is executed chunk by chunk interactively. It can be ran as `python zero_shot.py` with the
results grabbed from stdout.

