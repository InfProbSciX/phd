
import gzip
import pickle
import requests
import numpy as np
from io import BytesIO
import torch, torchvision
from sklearn.preprocessing import LabelEncoder

# mnist
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True)

x = np.concatenate((train_data.data, test_data.data), axis=0)
y = np.concatenate((train_data.targets, test_data.targets), axis=0)

x = x.reshape(x.shape[0], -1).astype(np.float32)

torch.save((x, y), 'mnist.pth')

# this code is from https://github.com/TorchDR/TorchDR/blob/e8f874dfaff647a7460d7c2248e75cc7131a9756/examples/single_cell/single_cell_readme.py
def download_and_load_dataset(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(BytesIO(response.content), "rb") as f:
        data = pickle.load(f)
    return data

url_macosko = "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz"
data_macosko = download_and_load_dataset(url_macosko)

x_macosko = data_macosko["pca_50"].astype("float32")
y_macosko = data_macosko["CellType1"].astype(str)
y_macosko_encoded = LabelEncoder().fit_transform(y_macosko)

x_macosko = x_macosko[y_macosko_encoded < 10]
y_macosko_encoded = y_macosko_encoded[y_macosko_encoded < 10]

torch.save((x_macosko, y_macosko_encoded), 'macosko.pth')

url_10x = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
data_10x = download_and_load_dataset(url_10x)

x_10x = data_10x["pca_50"].astype("float32")
y_10x = data_10x["CellType1"].astype("str")
y_10x_encoded = LabelEncoder().fit_transform(y_10x)

x_10x = x_10x[y_10x_encoded < 10]
y_10x_encoded = y_10x_encoded[y_10x_encoded < 10]

torch.save((x_10x, y_10x_encoded), 'zheng.pth')
