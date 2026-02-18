import os
import torch
import numpy as np
from tqdm import tqdm

data_list  = []
label_list = []
val_list = []

files = [f for f in os.listdir() if ('train_data_batch_' in f) or ('val_data.npz' in f)]
for f in tqdm(files):
    npz = np.load(f)
    data_list.append(npz["data"])
    label_list.append(npz["labels"])
    val_list.append(np.zeros_like(npz["labels"]) + (1 if f == 'val_data.npz' else 0))

all_data   = np.concatenate(data_list,  axis=0)
all_labels = np.concatenate(label_list, axis=0)
all_vals   = np.concatenate(val_list,   axis=0)

images_tensor = torch.from_numpy(all_data)
labels_tensor = torch.from_numpy(all_labels)
vals_tensor = torch.from_numpy(all_vals)

torch.save((images_tensor, labels_tensor, vals_tensor), "imagenet.pth")
