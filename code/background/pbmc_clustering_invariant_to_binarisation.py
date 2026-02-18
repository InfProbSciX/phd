
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt; plt.ion()

adata = sc.datasets.paul15()
adata.X = np.asarray(adata.X).astype(np.float32)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, target_sum=1e3)

sc.tl.pca(adata, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, random_state=0)
sc.tl.umap(adata, random_state=0)

sc.pl.umap(
    adata,
    color="paul15_clusters",
    title="before binarisation",
    legend_loc="on data",
    size=30,
)
plt.tight_layout()

adata_bin = adata.copy()
adata_bin.X[adata_bin.X > 0] = np.median(adata_bin.X[adata_bin.X > 0])

sc.tl.pca(adata_bin, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_bin, n_neighbors=15, n_pcs=50, random_state=0)
sc.tl.umap(adata_bin, random_state=0)

sc.pl.umap(
    adata_bin,
    color="paul15_clusters",
    title="after binarisation",
    legend_loc="on data",
    size=30,
)
plt.tight_layout()