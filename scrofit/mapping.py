from pulp import LpVariable, lpSum, LpProblem, LpStatus, LpMaximize, LpMinimize, value, PULP_CBC_CMD
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, triu, tril

import anndata as ad
import pandas as pd
import numpy as np
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

############### Spatial Distance #################

def knn_sparsify_aux(D, k):
    m, n = D.shape
    # Get the indices of the k-smallest values for each row
    indices = np.argsort(D, axis=1)[:, :k]
    # Create a sparse version of A by keeping only the k-smallest values
    rows = np.arange(D.shape[0])[:, None]  # Create row indices for broadcasting
    sparse_D = np.zeros_like(D)
    sparse_D[rows, indices] = D[rows, indices]
    return csr_matrix(sparse_D)

def knn_sparsify(XY_dist, n_neighbors=3,anchor='combine'):
    # knn sparfication
    row_sparse = knn_sparsify_aux(XY_dist, n_neighbors)
    col_sparse = knn_sparsify_aux(XY_dist.T, n_neighbors).T
    if anchor == 'combine':
        combined = row_sparse + col_sparse
    elif anchor == 'row':
        combined = row_sparse
    else:
        combined = col_sparse

    combined[combined > 0] = 1
    sparse = np.multiply(XY_dist, combined.todense())
    #print_msg(f'In the {anchor} sparse distance matrix,\n max value: {round(XY_dist.max(), 2)},\n min value: {round(XY_dist.min(), 2)},\n number of nonzero values: {combined.count_nonzero()}')
    return csr_matrix(sparse)


def estimate_radius(X_adata, spatial_type):
    X = X_adata[:100].obsm[spatial_type]
    D = euclidean_distances(X, X)  
    np.fill_diagonal(D, D.max())
    flattened = D.flatten()
    indices = np.argsort(flattened)[:10]
    top_min_values = flattened[indices]
    diameter = statistics.mode(top_min_values)
    return diameter/2

def adjusting_position(X_adata, Y_adata, 
                       ccs_type='spatial_align', raw_type='spatial',
                       mapping_method='MCMF', distance_method='euclidean'):
    #X_adata: source_adata, SM
    #Y_adata: target_data, ST  

    raw_y_radius = estimate_radius(Y_adata, raw_type)
    ccs_y_radius = estimate_radius(Y_adata, ccs_type)
    print(f'Estimate raw_y_radius {raw_y_radius}, ccs_y_radius {ccs_y_radius}')
    F = X_adata.obsm[f'mappingflow_{mapping_method}']
    XY_dist = X_adata.obsm[f'XY_spatial_{distance_method}_dist']
    Z = np.zeros(X_adata.obsm[ccs_type].shape)
    for u in tqdm(range(F.shape[1]), total=F.shape[1],
                  desc="---Adjusting source position within target's range"):   
        x_is, _ = F[:, u].nonzero()
        if x_is.shape[0] == 0:
              continue
        d = XY_dist[x_is, u].max()
        z_i = X_adata.obsm[ccs_type][x_is, :]
        raw_z_u = Y_adata.obsm[raw_type][u, :]
        z_u = Y_adata.obsm[ccs_type][u, :]
        if d > ccs_y_radius:
            raw_z_i = raw_z_u + raw_y_radius/d * (z_i - z_u)
        else:
            raw_z_i = raw_z_u + raw_y_radius/ccs_y_radius * (z_i - z_u)
        Z[x_is, :] = raw_z_i

    X_adata.obsm[f'spatial_{mapping_method}_map'] = Z


def calculate_spatial_distance(X_adata, Y_adata, 
                ccs_type='spatial_align',
                distance_method='euclidean'):
    XY_dist_name = f'XY_spatial_{distance_method}_dist'
    if XY_dist_name not in X_adata.obsm:
        X_coords = X_adata.obsm[ccs_type]
        Y_coords = Y_adata.obsm[ccs_type]
        XY_dist = cdist(X_coords, Y_coords, distance_method)
        X_adata.obsm[f'XY_spatial_{distance_method}_dist'] = XY_dist
    else:
        XY_dist = X_adata.obsm[f'XY_spatial_{distance_method}_dist'] 

    return XY_dist

# remove distant X source pixels
def remove_distant_kNN(X_adata, Y_adata, ccs_type='spatial_align', 
                       distance_method='euclidean', n_neighbors=3):
    #X_adata: source_adata, SM
    #Y_adata: target_data, ST
    print('-----')
    XY_dist = calculate_spatial_distance(X_adata, Y_adata, ccs_type=ccs_type, 
                                         distance_method=distance_method)
    XY_knn =  knn_sparsify(XY_dist, anchor='col', n_neighbors=n_neighbors)  
    xs, _ = XY_knn.nonzero()
    xs = list(set(xs))
    Z = np.empty((X_adata.shape[0],2))
    Z[:] = np.nan
    Z[xs, :] = X_adata.obsm[ccs_type][xs, :]
    X_adata.obsm[ccs_type+'_filtered'] = Z
    Y_adata.obsm[ccs_type+'_filtered'] = Y_adata.obsm[ccs_type]

def mapping_1NN(X_adata, Y_adata, 
                ccs_type='spatial_align',
                distance_method='euclidean'):
    #X_adata: source_adata, SM
    #Y_adata: target_data, ST
    XY_dist = calculate_spatial_distance(X_adata, Y_adata, ccs_type=ccs_type, 
                                         distance_method=distance_method)
    indices = np.argmin(XY_dist, axis=1)
    F = coo_matrix((np.ones(len(indices)), (range(len(indices)), indices))).tocsr()
    X_adata.obsm['mappingflow_1NN'] = F

      

def init_mdata(X_adata, Y_adata, source_key = 'SM', target_key = 'ST',
               layer='log1p', mapping_method='MCMF'):
    # X -> Y
    #X_adata: source_adata, SM, one
    #Y_adata: target_data, ST, multiple

    F = X_adata.obsm['mappingflow_MCMF']
    xs, ys = F.nonzero()
    #print('number of mappings', len(xs), len(ys))
    #print(source_key, xs.max())    
    #print(target_key, ys.max())
    #print(source_key, X_adata.layers[layer].max())  
    #print(target_key, Y_adata.layers[layer].max())  

    def _safe_array(X):
        if type(X) != np.ndarray:
            return X.todense()
        else:
            return X
        
    # merge layer
    layers = X_adata.layers.keys() & Y_adata.layers.keys()
    for i, layer in enumerate(layers):
        X = np.concatenate((
            _safe_array(X_adata.layers[layer][xs]), 
            _safe_array(Y_adata.layers[layer][ys])
        ), axis=1)
        if i == 0:
            mdata = ad.AnnData(csr_matrix(X))
        mdata.layers[f'{layer}'] = mdata.X

    # merge var
    var_df = pd.concat([X_adata.var, Y_adata.var])
    var_df['map_type'] = \
        [source_key] * X_adata.shape[1] + \
        [target_key] * Y_adata.shape[1]
    mdata.var = var_df

    # merge obs
    x_obs_df = X_adata[xs].obs.copy()
    x_obs_df = x_obs_df.reset_index()
    x_obs_df.columns = [f'{source_key}_{x}' for x in x_obs_df.columns]
    y_obs_df = Y_adata[ys].obs.copy()
    y_obs_df = y_obs_df.reset_index()
    y_obs_df.columns = [f'{target_key}_{y}' for y in y_obs_df.columns]
    obs_df = pd.concat([x_obs_df, y_obs_df], axis=1)
    obs_df[f'{source_key}_i'] = xs
    obs_df[f'{target_key}_i'] = ys
    mdata.obs = obs_df

    # merge obsm
    for key in X_adata.obsm.keys():
        mdata.obsm[f'{source_key}_{key}'] = X_adata[xs].obsm[key]
    for key in Y_adata.obsm.keys():
        mdata.obsm[f'{target_key}_{key}'] = Y_adata[ys].obsm[key]

    #print(X_adata)
    #print(Y_adata)
    #print(mdata)
    return mdata


###############ILP Solver#################
      

def mapping_MCMF(X_adata, Y_adata, ccs_type='spatial_align', 
                distance_method='euclidean', n_neighbors=3, 
                n_thread=4, alpha=1, beta=1, n_batch=1000,
                adata_layer = 'log1p', verbose=True):
    #X_adata: source_adata, SM
    #Y_adata: target_data, ST
    XY_dist = calculate_spatial_distance(X_adata, Y_adata, ccs_type=ccs_type, distance_method=distance_method)
    XY_knn =  knn_sparsify(XY_dist, anchor='row', n_neighbors=n_neighbors)  

    n_obs = X_adata.shape[0]
    batch_indices = np.array_split(np.arange(n_obs), np.ceil(n_obs / n_batch))
    F = np.zeros((X_adata.shape[0], Y_adata.shape[0]))
    i = 1
    for batch in tqdm(batch_indices, total=len(batch_indices),
                      desc='======Solving MCMF Mapping by ILP with batch'):
        print('---batch', i, len(batch))
        x_adata = X_adata[batch]
        X = x_adata.layers[adata_layer].toarray()
        XX_func_dist = euclidean_distances(X, X)  
        batch_f = mapping_ILP(XY_knn[batch, :], XY_dist[batch, :], XX_func_dist, n_thread=n_thread,
                    alpha=alpha, beta=beta, verbose=verbose)
        F[batch, :] = batch_f
        i += 1
    X_adata.obsm['mappingflow_MCMF'] = csr_matrix(F)


def mapping_ILP(XY_W, XY_dist, XX_func_dist, method='MCMF',
            n_thread=4, alpha=1, beta=1, verbose=True):
    
    prob = LpProblem(method, LpMaximize)

    sum_f, f_dict, n_dict = add_spatial_constraint(XY_W, XY_dist, prob, alpha=alpha)
    add_capacity_constraint(n_dict, prob)
 
    if beta > 0:
        sum_c, p_dict = add_func_constraint(XY_W, XX_func_dist, f_dict, prob, beta=beta)
        if verbose:
            print('p_u_ij num:', len(p_dict))

    t0 = time.time()
    status = prob.solve(PULP_CBC_CMD(threads=n_thread, msg=verbose))
    t1 = time.time()
    if verbose:
        print('LP solver time:', t1 - t0)    

    F = np.zeros(XY_W.shape)
    for i, u in tqdm(zip(*XY_W.nonzero()), 
                         desc='---Get the flow edge (unit: flow edge)',
                         total=XY_W.count_nonzero()):        
        f_name = f"f_i{i}_u{u}"
        v = value(f_dict[f_name])
        if v != 0: # and v is not None:  #???
            F[i, u] = v
    if verbose:
        print("Problem status: ", LpStatus[status])
        print("Total cost:", value(prob.objective))
        print("Spatial constraint cost:", value(sum_f))  
        if beta > 0:
            print("Functional constraint cost", value(sum_c))

    return F   
    
def add_spatial_constraint(
    XY_W, XY_dist, prob, alpha=1,
    method: str='MCMF', 
):
    sum_f = None
    f_dict, n_dict = {}, {}
    for i, u in tqdm(zip(*XY_W.nonzero()), 
                         desc='---Adding spatial constraints (unit: flow edge)',
                         total=XY_W.count_nonzero()):        
        c_iu = XY_dist[i, u]
    
        f_name = f"f_i{i}_u{u}"
        f_iu = LpVariable(f_name, cat='Binary')
        f_dict[f_name] = f_iu

        if sum_f is None:
            sum_f = f_iu * c_iu
        else:
            sum_f += f_iu * c_iu

        if i not in n_dict.keys():
            n_dict[i] = {'src': f_iu}
        else:
            n_dict[i]['src'] += f_iu

    prob += alpha*sum_f
    return sum_f, f_dict, n_dict

def add_func_constraint(
    XY_W, XX_func_dist, f_dict, prob,
    beta = 1,
    method: str='MCMF', 
):
    sum_c = None
    p_dict = {}
    us = XY_W.sum(axis=0).nonzero()[1]
    for u in tqdm(
        us, total=len(us),
        desc='---Adding functional constraints (unit: occupied target)', 
        ): 
        x_is, _ = XY_W[:, u].nonzero()
        for i, j in combinations(x_is, 2):      

            p_name = f"p_u{u}_i{i}_j{j}"
            p_uij = LpVariable(p_name, cat='Binary')
            p_dict[p_name] = p_uij

            d_ij = XX_func_dist[i, j]
            
            if sum_c is None:
                sum_c = p_uij * d_ij 
            else:
                sum_c += p_uij * d_ij    

            # linear constraint
            f_iu_name = f"f_i{i}_u{u}"
            f_iu = f_dict[f_iu_name]
            f_ju_name = f"f_i{j}_u{u}"
            f_ju = f_dict[f_ju_name]

            prob += p_uij <= f_iu
            prob += p_uij <= f_ju
            prob += p_uij >= f_iu + f_ju - 1

    prob += beta * sum_c
    return sum_c, p_dict


def add_capacity_constraint(
    n_dict: dict,
    prob
):
    for x_i, v in tqdm(n_dict.items(), total=len(n_dict),
                       desc='---Adding capacity constraints (unit: source)'):   
        for direction, sum_f in v.items():
            prob += sum_f == 1            
