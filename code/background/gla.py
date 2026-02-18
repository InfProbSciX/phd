
import numpy as np
from scipy.linalg import dft

np.random.seed(42)

n = 8; n_fft = 4; hop_length = 2
n_frames = n//hop_length - 1

D = dft(n_fft)

x = np.random.normal(size=n).reshape(-1, 1)

Dk = np.kron(np.eye(n_frames), D/n_fft**0.5)
Dhk = Dk.conj().T

W = np.zeros((n, n_frames))
for i in range(n_frames):
    W[(i*hop_length):(i*hop_length + n_fft), i] = 1

W /= (W @ np.ones((n_frames, 1)))

W_i = np.zeros((n, n_frames*n_fft))
for i in range(n_frames):
    idx_min, idx_max = i*hop_length, i*hop_length + n_fft
    W_i[idx_min:idx_max, (i*n_fft):((i + 1)*n_fft)] = np.diag(W[idx_min:idx_max, i])

W = np.ceil(W_i.T)

def stft(x):
    return Dk @ W @ x

def istft(stft_matrix):
    return W_i @ (Dhk @ S).real

S = stft(x)
# np.abs(istft(S) - x).max() # 1.7e-16
