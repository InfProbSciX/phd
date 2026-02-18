import matplotlib.pyplot as plt; plt.ion()
from skimage.filters import threshold_otsu, gaussian

img = plt.imread('CP-CC9-R3-15_I23_T0001F001L01A01Z01C01.tif')
img = img[256:-256, 256:-256]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img, cmap='GnBu')

img = gaussian(img, sigma=1.5)
thresh = threshold_otsu(img)

axs[1].imshow(img > thresh, cmap='GnBu')
axs[0].axis('off'); axs[1].axis('off')
plt.tight_layout()
