import squidpy as sq
import scanpy as sc
import anndata as ad
import numpy as np
import umap

def preprocess(adata, min_genes=10, min_cells=5):
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if 'log1p' not in adata.layers.keys():
        adata.layers['raw'] = adata.X
        sc.pp.log1p(adata)
        adata.layers['log1p'] = adata.X
