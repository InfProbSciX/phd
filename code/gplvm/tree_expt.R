
library(ggplot2)
library(magrittr)
library(data.table)
library(reticulate)
library(SingleCellExperiment)

use_python('/Users/aditya/miniconda3/bin/ipython', required=T)

set.seed(42)

# https://biocellgen-public.svi.edu.au/mig_2019_scrnaseq-workshop/trajectory-inference.html
# https://oncoscape.v3.sttrcancer.org/atlas.gs.washington.edu.mouse.rna/downloads
data = readRDS('epithelial_cds.rds')

idx = 1:dim(data)['Samples']

subsetter = function(name, n_samp_min=1000) {
    subset = idx[data$subtrajectory_name == name]
    n_samp = as.integer(length(subset)*0.25)
    if(n_samp < n_samp_min & n_samp_min <= length(subset))
        n_samp = n_samp_min
    return(sample(subset, n_samp))
}

idx = unique(data$subtrajectory_name) %>%
      sapply(subsetter) %>%
      unlist %>% unname %>% sort

mat = log1p(t(exprs(data))[idx, ] / sizeFactors(data)[idx])
sds = apply(mat, 2, sd)
mat = mat[, unname(which(sds > quantile(sds, 0.9)))]

py_run_string('import numpy as np')
py$np$save('mat.npy', mat)

```{python}

import numpy as np
from umap import UMAP

mat = np.load('mat.npy', allow_pickle=True).item().toarray()

model = UMAP(verbose=True, output_metric='hyperboloid', random_state=42).fit(mat)
embedding = model.embedding_ # transform(r.mat)

np.save('emb.npy', embedding)

# from https://umap-learn.readthedocs.io/en/latest/embedding_space.html#bonus-embedding-in-hyperbolic-space

def coords(embedding):
    import numpy as np
    z = np.sqrt(1 + np.sum(embedding**2, axis=1))
    disk = embedding / (1 + z.reshape(-1, 1))
    return disk

disk = coords(embedding)
np.save('disk.npy', disk)

```

py_run_string('disk = np.load("disk.npy")')
py_run_string('embedding = np.load("emb.npy")')

plot_df = data.table(day = data$day[idx],
                     subtrajectory = data$subtrajectory_name[idx])

plot_df[, disk_x := py$disk[, 1]]
plot_df[, disk_y := py$disk[, 2]]

plot_df[, x := py$embedding[, 1]]
plot_df[, y := py$embedding[, 2]]

cols = colorRampPalette(RColorBrewer::brewer.pal(12, 'Paired'))(
    plot_df$subtrajectory %>% unique %>% length
)

gridExtra::grid.arrange(grobs=list(

    ggplot(plot_df, aes(x=x, y=y, color=subtrajectory)) +
        geom_point(alpha=0.05) +
        scale_color_manual(values=cols) +
        theme_classic() +
        guides(colour=guide_legend(override.aes=list(alpha=1), byrow=T)) +
        theme(legend.text=element_text(size=6),
              legend.spacing.y=unit(-0.1, 'cm')) +
        labs(x='Latent Variable 1 (Intrinsic Coords)',
             y='Latent Variable 2 (Intrinsic Coords)',
             title='LVs by Trajectories/Cell Types'),

    ggplot(plot_df) +
        geom_point(aes(x=disk_x, y=disk_y, color=day), alpha=0.5) +
        geom_path(aes(x=x, y=y), data=data.table(
            x=cos(seq(0, 2*pi, l=1000)),
            y=sin(seq(0, 2*pi, l=1000))
        )) +
        scale_color_brewer(palette='Spectral') +
        theme_classic() +
        guides(colour=guide_legend(override.aes=list(alpha=1))) +
        labs(x='Latent Variable 1 (Disk Coords)',
             y='Latent Variable 2 (Disk Coords)',
             title='LVs by Collection Time')
), ncol=2)
