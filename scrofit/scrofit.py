from itertools import combinations
import os
import math
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


import scrofit.preprocessing as pp
from scrofit.alignment import align_slides
import scrofit.plotting as pl
from scrofit.util import print_msg
from scrofit.mapping import mapping_MCMF, mapping_1NN, adjusting_position, remove_distant_kNN
import scrofit.embedding as embed


class SCRoFit(object):

    def __init__(self, adata_dict=None, out_dir='./', sample='sample'):
        self.adata_dict = adata_dict
        self.out_dir = out_dir
        self.sample = sample
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        for key, adata in adata_dict.items():
            if key == 'ST':
                pp.preprocess(adata, min_genes=100, min_cells=5)
            if key.startswith('SM'):
                pp.preprocess(adata, min_genes=100, min_cells=5)

    def check_slides(self, s=1, ncol=3, nrow=2, figsize=(10, 8), fig_fn=None):
        pl.plot_CCS(self, s=s, ncol=ncol, nrow=nrow, figsize=figsize, fig_fn=fig_fn)

    def flip_slides(self, key, direction='hv'):
        if 'v' in direction:
            max = self.adata_dict[key].obsm['spatial'][:, 1].max()
            self.adata_dict[key].obsm['spatial'][:, 1] = max - self.adata_dict[key].obsm['spatial'][:, 1]
        if 'h' in direction:
            max = self.adata_dict[key].obsm['spatial'][:, 0].max()
            self.adata_dict[key].obsm['spatial'][:, 0] = max - self.adata_dict[key].obsm['spatial'][:, 0]
        #pl.plot_CCS(self)

    def align_slides(self, anchor_key='ST', method='rir', figsize=(10, 8),
                     echo=True, plot=True):
        if echo:
            print(f'SCRoFit aligning slides to {anchor_key} with {method}...')

        if 'spatial_' + method not in self.adata_dict[anchor_key].obsm.keys():
            align_slides(self.adata_dict, anchor_key=anchor_key, method=method)

        if plot:
            fig_fn = f'{self.out_dir}/{self.sample}_alignment.png'
            pl.plot_CCS(self, figsize=figsize, fig_fn=fig_fn)

    def mapping(self, source_key='SM', target_key='ST', mapping_method='MCMF',
                ccs_type='spatial_align', distance_method='euclidean', 
                n_neighbors=3, filter_target=True,
                n_thread=4, alpha=1, beta=1, n_batch=1000,
                adata_layer='log1p', verbose=True):
        X_adata = self.adata_dict[source_key]
        Y_adata = self.adata_dict[target_key]
        # mapping X (source) to Y (target)
        if filter_target:
            # remove distant X source pixels
            remove_distant_kNN(X_adata, Y_adata, ccs_type=ccs_type, 
                    distance_method=distance_method, n_neighbors=n_neighbors)               
            ccs_type += '_filtered'

        if mapping_method == '1NN':
            mapping_1NN(X_adata, Y_adata, ccs_type=ccs_type, 
                        distance_method=distance_method
                        )        
        if mapping_method == 'MCMF':
            mapping_MCMF(X_adata, Y_adata, ccs_type=ccs_type, 
                         n_neighbors=n_neighbors, 
                    n_thread=n_thread, n_batch=n_batch, alpha=alpha, beta=beta,
                    adata_layer=adata_layer, verbose=verbose)
            
        adjusting_position(X_adata, Y_adata, mapping_method=mapping_method, ccs_type=ccs_type)

    def embedding(self, source_key='ST', target_key='SM', 
                  mapping_method='MCMF', verbose=True):

        embed.embedding(X_adata, Y_adata, mapping_method)



    def transfer_anno(self, headers, source_key='ST', target_key='SM', 
                      mapping_method='MCMF',
                      verbose=True):

        X_adata = self.adata_dict[source_key]
        Y_adata = self.adata_dict[target_key]    
        F = Y_adata.obsm[f'mappingflow_{mapping_method}']

        ys, xs = F.nonzero()
        cols = list(set(headers) & set(X_adata.obs.columns))
        df = X_adata[xs, :].obs[cols].copy()
        df['ST_id'] = df.index
        df.index = Y_adata[ys, :].obs.index
        for x in df.columns:
            col = source_key + '_' + x 
            Y_adata.obs[col] = df[x].to_dict()

        cols = list(set(headers) & set(X_adata.var_names))
        for col in cols:
            df[col] = X_adata[xs, col].X.todense()
            Y_adata.obs[col] = df[col].to_dict()
            