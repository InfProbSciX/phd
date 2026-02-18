import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from scipy.io import loadmat

m = loadmat("Downloads/Hafting_Fig2d_Trial1.mat")

x = np.asarray(m["pos_x"], dtype=float).ravel()
y = np.asarray(m["pos_y"], dtype=float).ravel()
t = np.asarray(m["pos_timeStamps"], dtype=float).ravel()

spk_t = np.asarray(m["rat11015_t1c1"], dtype=float).ravel()

ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
x, y, t = x[ok], y[ok], t[ok]

spk_x = np.interp(spk_t, t, x)
spk_y = np.interp(spk_t, t, y)

bin_size = 5.0   # in same units as pos_x/pos_y
sigma = .2      # smoothing in bins
occ_thr = 0.10   # seconds; drop bins visited less than this

xbins = np.arange(x.min(), x.max() + bin_size, bin_size)
ybins = np.arange(y.min(), y.max() + bin_size, bin_size)

dt = np.diff(t)
dt = np.append(dt, np.median(dt))  # last sample duration

occ, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], weights=dt)
spk, _, _ = np.histogram2d(spk_x, spk_y, bins=[xbins, ybins])

rate = spk / (occ + 1e-12)
rate[occ < occ_thr] = np.nan

plt.figure(figsize=(5, 4.5))
plt.imshow(
    rate.T,
    origin="lower",
    aspect="equal",
    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
)
plt.axis('off')
plt.tight_layout()
