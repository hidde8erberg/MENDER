import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import MENDER
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysodb
from sklearn.metrics import *
import time 

#NOTE for runtime reasons, this code is run for 1 slice per age category only --> the numbers will not directly match the figure in the paper. 
#Whenever the full dataset is used (31 slices), my computer crashes. 

if __name__ == '__main__':

    n_cores = os.cpu_count()  # check how many cores you have
    print(n_cores)            # e.g. 4 or 8 on a typical laptop

    if os.path.exists('Allen2022_MERFISH_aging.h5ad'):
        # Fast path: load from disk
        adata_raw = sc.read_h5ad('Allen2022_MERFISH_aging.h5ad')
    else:
        # Slow path: fetch from SODB and save
        sodb = pysodb.SODB()
        adata_dict = sodb.load_dataset('Allen2022Molecular_aging')

        adata_list = []
        for si in adata_dict.keys():
            adata = adata_dict[si]
            adata.obs['slice_id'] = si
            adata_list.append(adata)

        adata_raw = adata_list[0].concatenate(adata_list[1:])
        adata_raw.obs['slice_id'] = adata_raw.obs['slice_id'].astype('category')
        adata_raw.obs['gt'] = adata_raw.obs['tissue'].astype('category')
        adata_raw.obs['ct'] = adata_raw.obs['clust_annot'].astype('category')

        adata_raw.write_h5ad('Allen2022_MERFISH_aging.h5ad')

# print(adata_raw.obs.columns.tolist())
# print(adata_raw.obs['age'].unique())
# print(adata_raw.obs.groupby('slice_id')['age'].first().sort_values())

#! tot hier, fetch data first

# Map slice IDs to aging stages
# print(adata_raw.obs.groupby('slice_id')['age'].first().sort_values())
    slice_4wk  = 'MsBrainAgingSpatialDonor_4_2'   # one 4-week slice
    slice_24wk = 'MsBrainAgingSpatialDonor_10_0'   # one 24-week slice
    slice_90wk = 'MsBrainAgingSpatialDonor_3_0'   # one 90-week slice

    keep = [slice_4wk, slice_24wk, slice_90wk]
    adata_sub = adata_raw[adata_raw.obs['slice_id'].isin(keep)].copy()
    adata_sub.obs['slice_id'] = adata_sub.obs['slice_id'].astype('category')
    adata_sub.obs['stage'] = adata_sub.obs['age'].astype('category')

#run MENDER 
    adata = adata_sub.copy()
    batch_obs = 'slice_id'
    gt_obs = 'gt'
    scale = 6
    radius = 15
    time_start = time.time()
    n_cls = np.unique(adata_sub.obs[gt_obs]).shape[0]

    msm = MENDER.MENDER(
        adata,
        batch_obs=batch_obs,
        ct_obs='ct',
        random_seed=0
    )

    msm.prepare()
    msm.set_MENDER_para(
        n_scales=scale,
        nn_mode='radius',
        nn_para=radius,
    )
    msm.run_representation_mp(n_cores-2)

# Sub-clustering to get 9 domains (as in Fig. 5A–H)
    msm.run_clustering_normal(-0.5)

    time_cost = time.time()-time_start
    print(f'this run took {time_cost} seconds for {adata_sub.shape[0]} cells')

    adata_mender = msm.adata_MENDER.copy()
    msm.adata_MENDER.write_h5ad('Allen2022_MERFISH_aging_MENDER.h5ad')
    adata_mender.obs['stage'] = adata_sub.obs['stage'].values

    # now plot 
    # Color palette matching the paper (3 stages)
    stage_order = ['4wk', '24wk', '90wk']   # adjust to match actual values
    stage_colors = {
        '4wk':  '#4393C3',
        '24wk': '#F4A582',
        '90wk': '#D6604D',
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    for ax, domain_col, title in zip(
        axes,
        ['gt', 'MENDER'],
        ['Original annotation', 'MENDER']
    ):
        # Build a cross-tabulation: domain × stage (cell counts)
        ct = pd.crosstab(
            adata_mender.obs[domain_col],
            adata_mender.obs['stage']
        )
        # Normalize to proportions per domain (row-normalize)
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        # Reorder columns to match stage order
        ct_norm = ct_norm[[s for s in stage_order if s in ct_norm.columns]]
        
        # Stacked bar plot
        bottom = np.zeros(len(ct_norm))
        for stage in stage_order:
            if stage not in ct_norm.columns:
                continue
            ax.bar(
                range(len(ct_norm)),
                ct_norm[stage].values,
                bottom=bottom,
                color=stage_colors[stage],
                label=stage,
                edgecolor='white',
                linewidth=0.5
            )
            bottom += ct_norm[stage].values
        
        ax.set_xticks(range(len(ct_norm)))
        ax.set_xticklabels(ct_norm.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(title='Stage', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure5H.png', dpi=200, bbox_inches='tight')
    plt.show()