import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import MENDER
from MENDER.utils import compute_NMI, compute_ARI
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysodb
from sklearn.metrics import *
import time 

if __name__ == '__main__': 
  n_cores = os.cpu_count()

  # import raw data 
  adata_raw = sc.read_h5ad('../subfigure/Allen2022_MERFISH_aging.h5ad')
  print(f'all columns in raw data: {adata_raw.obs.columns.tolist()} \n')
  print(f'raw data age categories: {adata_raw.obs["age"].unique().tolist()} \n')
  print(f'raw data cluster annotation categories: {adata_raw.obs["clust_annot"].unique().tolist()} \n')
  print(f'raw data slice numbers: {adata_raw.obs["slice"].unique().tolist()} \n')
  print(f'raw data organism: {adata_raw.obs["organism"].unique().tolist()} \n')
  print(f'raw data tissue categories: {adata_raw.obs["tissue"].unique().tolist()} \n')
  print(f'raw data cell type categories: {adata_raw.obs["cell_type_annot"].unique().tolist()} \n')

  print(f'This raw anndata contains {adata_raw.shape[0]} cells and {len(adata_raw.obs["slice_id"].unique().tolist())} batches')

  # select 1 slice subset for faster processing using different resolutions
  slice_4wk  = 'MsBrainAgingSpatialDonor_4_2'   # one 4-week slice
  # slice_24wk = 'MsBrainAgingSpatialDonor_10_0'   # one 24-week slice
  # slice_90wk = 'MsBrainAgingSpatialDonor_3_0'   # one 90-week slice

  # keep = [slice_4wk, slice_24wk, slice_90wk]
  keep = [slice_4wk]
  adata_sub = adata_raw[adata_raw.obs['slice_id'].isin(keep)].copy()
  adata_sub.obs['slice_id'] = adata_sub.obs['slice_id'].astype('category')
  adata_sub.obs['stage'] = adata_sub.obs['age'].astype('category')
  print(f'This subset data contains {adata_sub.shape[0]} cells and {len(adata_sub.obs["slice_id"].unique().tolist())} batches')

  # --------------------------------------- RUN mender only if resolution files don't exist yet ---------------------------------------
  # these resolutions are directly passed to the scanpy.tl.leiden funcs as Leiden resolutions. They are first turned positive (res = -target) before passed the function. Higher values lead to more clusters. 
  resolutions = [-0.1, -0.3, -0.5, -0.7, -1.0, -1.5, -2.0, -2.5]

  for i in resolutions: 
    if os.path.exists(f'{slice_4wk}_res_{i}.h5ad'):
      print(f'MENDER output data for res {i} already exists! \n')
    else: 
      print(f'MENDER output data for res {i} does not exist, running MENDER now \n')

  # first do the basic 1 time processing stuff 
  adata = adata_sub.copy()
  batch_obs = 'slice_id'
  gt_obs = 'gt'
  scale = 6
  radius = 15
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

  print(f'Number of ground truth domains: {n_cls}')
  results = []

  # now actually run mender and save to files 
  for i in resolutions: 
    if i == -0.5: 
      # ground truth needs to be printed once. 
      msm.output_cluster_all(obs=gt_obs, obs_gt=None)
      plt.suptitle('Ground truth', fontsize=12)
      plt.savefig('resolution_ground_truth.png', dpi=150, bbox_inches='tight')
      plt.close('all')

    # refresh and rerun clustering with different leiden resolution 
    msm.refresh_adata_MENDER() 
    time_start = time.time() 
    msm.run_clustering_normal(i)
    time_cost = time.time()-time_start
    print(f'this run with resolution {i} took {time_cost} seconds for {adata_sub.shape[0]} cells')

    # compute comparison metrics and add to results dict 
    adata_mender = msm.adata_MENDER.copy()  
    domains_mender = adata_mender.obs['MENDER'].nunique()
    msm.adata_MENDER.write_h5ad(f'{slice_4wk}_res_{i}.h5ad')
    nmi = compute_NMI(adata_mender, gt_obs, 'MENDER')
    ari = compute_ARI(adata_mender, gt_obs, 'MENDER')
    results.append({
            'resolution': i,
            'n_domains': domains_mender,
            'NMI': nmi,
            'ARI': ari,
            'time_s': time_cost
        })
    msm.output_cluster_all(obs='MENDER', obs_gt=gt_obs)
    plt.suptitle(f'res={i}', fontsize=12, y = 1.0)
    plt.savefig(f'resolution_mender_res{i}.png', dpi=150, bbox_inches='tight')
    plt.close('all')

  # save results to csv 
  results_df = pd.DataFrame(results)
  results_df.to_csv('resolution_analysis.csv', index=False)

  # final summary comparison 
  fig, ax1 = plt.subplots(figsize=(8, 4))
  ax2 = ax1.twinx()
  ax1.plot(results_df['resolution'], results_df['NMI'], 'o-', color='steelblue', label='NMI')
  ax1.plot(results_df['resolution'], results_df['ARI'], 's--', color='cornflowerblue', label='ARI')
  ax2.plot(results_df['resolution'], results_df['n_domains'], '^-', color='tomato', label='# domains')
  ax1.set_xlabel('Leiden Resolution')
  ax1.set_ylabel('NMI / ARI', color='steelblue')
  ax2.set_ylabel('Number of domains', color='tomato')
  ax1.legend(loc='upper left')
  ax2.legend(loc='upper right')
  plt.title('Effect of resolution on MENDER domain identification')
  plt.tight_layout()
  plt.savefig('resolution_summary.png', dpi=150, bbox_inches='tight')
  plt.show()