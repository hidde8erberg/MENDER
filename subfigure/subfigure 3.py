import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import MENDER
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import time

# Load data from local CSVs
data_dir = os.path.dirname(os.path.abspath(__file__))

# Expression matrix: row 0 is a numeric index header, row 1 has gene names
expr = pd.read_csv(os.path.join(data_dir, 's3_cell_by_gene.csv'), skiprows=1, index_col=0)
expr.index = expr.index.astype(str)

# Cell metadata: contains spatial coords and layer (cell type) labels
meta = pd.read_csv(os.path.join(data_dir, 's3_mapped_cell_table.csv'), index_col=0)
meta.index = meta['sample_name'].astype(str)

# Align to common cells
common = expr.index.intersection(meta.index)
expr = expr.loc[common]
meta = meta.loc[common]

# Build AnnData
adata = sc.AnnData(
    X=expr.values.astype(float),
    obs=meta,
    var=pd.DataFrame(index=expr.columns)
)
adata.obsm['spatial'] = meta[['x', 'y']].values
adata.obs['ct_sub'] = meta['layer'].astype(str)

adata_dict = {'s3': adata}


# specify slice_id and ground truth
adata_list = []
for si in adata_dict.keys():
    adata = adata_dict[si]
    adata.obs['slice_id'] = si
    adata_list.append(adata)
adata_raw = adata_list[0].concatenate(adata_list[1:]) if len(adata_list) > 1 else adata_list[0].copy()
adata_raw.obs['slice_id'] = adata_raw.obs['slice_id'].astype('category')
adata_raw.obs['ct'] = adata_raw.obs['ct_sub'].astype('category')

print(adata_raw)

sc.pl.embedding(adata_raw, basis='spatial', color='layer', show=False)
plt.savefig(os.path.join(data_dir, 'spatial_layer.png'), dpi=150, bbox_inches='tight')
plt.show()


batch_obs = 'slice_id'
gt_obs = 'layer'

# input parameters of MENDER
scale = 6
radius = 15

# estimate number of domains
n_cls = np.unique(adata.obs[gt_obs]).shape[0]


# record running time
time_st = time.time()


adata = adata_raw.copy()



######### determine cell state using standard Leiden [start]  #########
# this step can be optionally skipped if reliable cell type annotation is available
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata,resolution=2,key_added='ct',random_state=666)
adata.obs[batch_obs] = adata.obs[batch_obs].astype('category')
adata.obs['ct'] = adata.obs['ct'].astype('category')
######### determine cell state using standard Leiden [end]  #########


# main body of MENDER
msm = MENDER.MENDER(
    adata,
    batch_obs = batch_obs,
    # determine which cell state to use
    # we use the cell state got by Leiden
    ct_obs='ct',
    random_seed=666
)


# set the MENDER parameters


msm.prepare()
msm.set_MENDER_para(
    # default of n_scales is 6
    n_scales=scale,

    # for single cell data, nn_mode is set to 'radius'
    nn_mode='radius',

    # default of n_scales is 15 um (see the manuscript for why).
    # MENDER also provide a function 'estimate_radius' for estimating the radius
    nn_para=radius,

)
# construct the context representation
msm.run_representation()

# set the spatial clustering parameter
# positive values for the expected number of domains
# negative values for the clustering resolution
msm.run_clustering_normal(n_cls)

time_ed = time.time()
time_cost = time_ed-time_st

# the plot function has two parameters
# obs: the observation to plot
# gt_obs: the ground truth observation to compute NMI and ARI, can be set to None if not available
msm.output_cluster_all(obs='MENDER',obs_gt=gt_obs)
print('MENDER prediction')

# the plot function has two parameters
# obs: the observation to plot
# gt_obs: the ground truth observation to compute NMI and ARI, can be set to None if not available

msm.output_cluster_all(obs=gt_obs,obs_gt=None)
print('ground truth')

print(f'running time: {time_cost} s')