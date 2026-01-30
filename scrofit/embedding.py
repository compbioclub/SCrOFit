from pulp import LpVariable, lpSum, LpProblem, LpStatus, LpMaximize, LpMinimize, value, PULP_CBC_CMD
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, triu, tril

import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import time
from multiprocessing import Pool
from functools import partial
import networkx as nx
from itertools import combinations
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix, lil_matrix
import statistics

from scrofit.util import print_msg, get_array


def embedding(mdata, source_key='SM', target_key='ST', 
              mapping_method = 'MCMF',
              layer='log1p'):
    # source is SM, target is ST
    # X -> Y

    sc.pp.pca(mdata)
    sc.pp.neighbors(mdata)
    sc.tl.umap(mdata)

    pass