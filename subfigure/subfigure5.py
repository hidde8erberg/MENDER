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

#NOTE for runtime reasons, this code can be run for 1 slice per age category only --> the numbers will not directly match the figure in the paper. 

if __name__ == '__main__':

# ----------------------------------- RUN below here to retrieve data first, either from file or from the sodb connection ------------------------------

    n_cores = os.cpu_count()  # check how many cores you have
    print(f'Number of cores available: {n_cores}\n')            # e.g. 4 or 8 on a typical laptop

    if os.path.exists('Allen2022_MERFISH_aging.h5ad'):
        print(f'Raw AnnData already exists \n')
        adata_raw = sc.read_h5ad('Allen2022_MERFISH_aging.h5ad')
    else:
        print(f'Raw AnnData does not exist, currently fetching from SODB \n')
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
        print(f'Raw AnnData written to file \n')

    print(f'all columns in raw data: {adata_raw.obs.columns.tolist()} \n')
    print(f'raw data age categories: {adata_raw.obs["age"].unique().tolist()} \n')
    print(f'raw data cluster annotation categories: {adata_raw.obs["clust_annot"].unique().tolist()} \n')
    print(f'raw data slice numbers: {adata_raw.obs["slice"].unique().tolist()} \n')
    print(f'raw data organism: {adata_raw.obs["organism"].unique().tolist()} \n')
    print(f'raw data tissue categories: {adata_raw.obs["tissue"].unique().tolist()} \n')
    print(f'raw data cell type categories: {adata_raw.obs["cell_type_annot"].unique().tolist()} \n')
    print(adata_raw.obs.groupby('slice_id')['age'].first().sort_values())
    print('\n') 

    # convert age to stage 
    adata_raw.obs['stage'] = adata_raw.obs['age'].astype('category')

# ------------------------------------------ RUN code below to get only a subset of the raw data slices (for faster plotting) ------------------------------
    # gebruik 1 slice per age voor sneller analyse 
    # print(adata_raw.obs.groupby('slice_id')['age'].first().sort_values())
    # slice_4wk  = 'MsBrainAgingSpatialDonor_4_2'   # one 4-week slice
    # slice_24wk = 'MsBrainAgingSpatialDonor_10_0'   # one 24-week slice
    # slice_90wk = 'MsBrainAgingSpatialDonor_3_0'   # one 90-week slice

    # keep = [slice_4wk, slice_24wk, slice_90wk]
    # adata_sub = adata_raw[adata_raw.obs['slice_id'].isin(keep)].copy()
    # adata_sub.obs['slice_id'] = adata_sub.obs['slice_id'].astype('category')
    # adata_sub.obs['stage'] = adata_sub.obs['age'].astype('category')


#-------------------------------------------- RUN Mender (if necessary) and plot results) -------------------------------------------------------------------
#TODO: only if the mender h5ad doesn't exist yet! 
#TODO: make decision tree for usage of all vs just 1 slice per time stage (save two different h5ad data outputs MENDER)

    if os.path.exists('Allen2022_MERFISH_aging_MENDER_FULL.h5ad'):
        print('Full MENDER file already exists! retrieving from file \n')
        adata_mender = sc.read_h5ad('Allen2022_MERFISH_aging_MENDER_FULL.h5ad')

    elif os.path.exists('Allen2022_MERFISH_aging_MENDER_sub.h5ad'): 
        print('Subset of MENDER results already exist! retrieving from file \n')
        adata_mender = sc.read_h5ad('Allen2022_MERFISH_aging_MENDER_sub.h5ad')

    else: 
        print('MENDER output data does not exist yet! Will run MENDER now \n')

        adata = adata_raw.copy()
        batch_obs = 'slice_id'
        gt_obs = 'gt'
        scale = 6
        radius = 15
        time_start = time.time()
        n_cls = np.unique(adata_raw.obs[gt_obs]).shape[0]

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

        # clustering to get 9 domains as shown in tutorial 
        msm.run_clustering_normal(-0.5)

        time_cost = time.time()-time_start
        print(f'this run took {time_cost} seconds for {adata_raw.shape[0]} cells')

        adata_mender = msm.adata_MENDER.copy()
        if len(adata_raw.obs["slice_id"].unique().tolist()) == 31:  # There are 31 batches in the original data. Save in respective anndata files. 
            msm.adata_MENDER.write_h5ad('Allen2022_MERFISH_aging_MENDER_FULL.h5ad')
            print(f'Full MENDER results written to file! \n')
        else: 
            msm.adata_MENDER.write_h5ad('Allen2022_MERFISH_aging_MENDER_sub.h5ad')
            print(f'Subset of MENDER results written to file! \n')

    #---------------------------------------------------------- plotting done from here -------------------------------------------------------------------------
    print(f'Adding stage to mender results  \n')
    adata_mender.obs['stage'] = adata_raw.obs['stage'].values 
    
    # Color palette matching stages in paper
    stage_order = ['90wk', '24wk', '4wk']
    stage_colors = {
        '90wk': '#DF73FF',
        '24wk': '#F2B949',
        '4wk':  '#568203',
    }
    gt_order = ['brain ventricle', 'corpus callosum', 'cortical layer II/III', 'cortical layer V', 'cortical layer VI', 'olfactory region', 'pia mater', 'striatum'] 

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    for ax, domain_col, title in zip(
        axes,
        ['gt', 'MENDER'],
        ['Stage distribution across domains (original annotation)', 'Stage distribution across domains (MENDER)']
    ):
        # Build a cross-tabulation: domain × stage (cell counts)
        ct = pd.crosstab(
            adata_mender.obs[domain_col],
            adata_mender.obs['stage']
        )
        # Normalize to frequencies per domain 
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        # Reorder columns to match stage order
        ct_norm = ct_norm[[s for s in stage_order if s in ct_norm.columns]]
        
        #reorder bars to match the original paper 
        if domain_col == 'gt':
            ct_norm = ct_norm.reindex([d for d in gt_order if d in ct_norm.index])
        
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
        ax.set_ylabel('Domain Frequency', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(title='Stage', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure5H.png', dpi=200, bbox_inches='tight')
    plt.show()