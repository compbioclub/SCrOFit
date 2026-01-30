import numpy as np
import anndata as ad
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
import random
import pandas as pd


def evaluate_alignment(adata_dict, anchor_key=None, align_method='icp', evaluate_method='mse'):
    # Valid align_methods and evaluate_methods
    valid_align_methods = ['scaling', 'icp', 'rir', 'convex', 'pca']
    valid_evaluate_methods = ['mse', 'appd', 'hausdorff', 'rmsd', 'chamfer']

    # Ensure align_method and evaluate_method are lists
    align_method = [align_method] if isinstance(align_method, str) else align_method
    evaluate_method = [evaluate_method] if isinstance(evaluate_method, str) else evaluate_method


    # Validate align_method
    for method in align_method:
        if method not in valid_align_methods:
            raise ValueError(f"Unknown method: {method}, please set the method to be one of {valid_align_methods}.")

    # Validate evaluate_method
    for method in evaluate_method:
        if method not in valid_evaluate_methods:
            raise ValueError(f"Unknown evaluation method: {method}, please set the method to be one of {valid_evaluate_methods}.")

    res_list = []
    for key in adata_dict.keys():
        if key == anchor_key:
            continue

        for align in align_method:
            for evaluate in evaluate_method:
                res = compare_anchor_target(adata_dict, anchor_key, key, align, evaluate)
                res_list.append(res)
    
    return pd.DataFrame(res_list)

###### helper function #######  

def compare_anchor_target(adata_dict, anchor_key, key, align_method, evaluate_method):
    # Compare the spatial alignment between the anchor and target datasets
    res = {'anchor': anchor_key, 'target': key, 'align_method': align_method}
    
    anchor_points = adata_dict[anchor_key].obsm[str('spatial_'+align_method)]
    target_points = adata_dict[key].obsm[str('spatial_'+align_method)]
    
    if evaluate_method == 'hausdorff':
        # Hausdorff Distance
        res['evaluate_method'] = evaluate_method
        res['eval_score'] = evaluation_hausdorff(anchor_points, target_points, anchor_key, key)
        return res
    
    # The following evaluation methods require pairwise points
    anchor_points, target_points = get_pairwise_points(adata_dict[anchor_key].obsm[str('spatial_'+align_method)], 
                                                       adata_dict[key].obsm[str('spatial_'+align_method)])
    
    if evaluate_method == 'mse':
        # Mean Squared Error
        res['evaluate_method'] = evaluate_method
        res['eval_score'] = evaluation_mse(anchor_points, target_points, anchor_key, key)
    elif evaluate_method == 'appd':
        # Average Point-to-Point Distance
        res['evaluate_method'] = evaluate_method
        res['eval_score'] = evaluation_appd(anchor_points, target_points, anchor_key, key)
    elif evaluate_method == 'rmsd':
        # Root Mean Square Deviation
        res['evaluate_method'] = evaluate_method
        res['eval_score'] = evaluation_rmsd(anchor_points, target_points, anchor_key, key)
    elif evaluate_method == 'chamfer':
        # Chamfer Distance
        res['evaluate_method'] = evaluate_method
        res['eval_score'] = evaluation_chamfer(anchor_points, target_points, anchor_key, key)
    return res

def get_pairwise_points(anchor_points, target_points):
    # Ensure both point clouds have the same size for comparison
    # Efficiently generate pairwise points using broadcasting (cartesian product)
    '''
    Generate pairwise points from two sets of points (anchor_points and target_points)
    by calculating the Cartesian product of the two sets.

    Parameters:
    anchor_points (np.ndarray): A NumPy array of shape (N, 2) representing N anchor points.
    target_points (np.ndarray): A NumPy array of shape (M, 2) representing M target points.

    Returns:
    tuple: Two NumPy arrays representing the new anchor points and target points after
    generating the pairwise points.
    '''
    anchor_shape = anchor_points.shape[0]
    target_shape = target_points.shape[0]

    expanded_anchor_points = anchor_points[:, np.newaxis, :]
    expanded_target_points = target_points[np.newaxis, :, :]

    broadcasted_anchor_points = np.broadcast_to(expanded_anchor_points, (anchor_shape, target_shape, 2))
    broadcasted_target_points = np.broadcast_to(expanded_target_points, (anchor_shape, target_shape, 2))

    new_anchor_points = broadcasted_anchor_points.reshape(-1, 2)
    new_target_points = broadcasted_target_points.reshape(-1, 2)

    return new_anchor_points, new_target_points


def evaluation_mse(anchor_points, target_points, anchor_key, key):
    value = mean_squared_error(anchor_points, target_points)
    print(f"Mean Squared Error between {anchor_key} and {key}: {value}")
    return value

def evaluation_appd(anchor_points, target_points, anchor_key, key):
    value = np.mean(np.linalg.norm(anchor_points - target_points, axis=1))
    print(f"Average Point-to-Point Distance between {anchor_key} and {key}: {value}")
    return value
            
def evaluation_hausdorff(anchor_points, target_points, anchor_key, key):
    forward_hausdorff = directed_hausdorff(target_points, anchor_points)[0]
    backward_hausdorff = directed_hausdorff(anchor_points, target_points)[0] 
    value = max(forward_hausdorff, backward_hausdorff)
    print(f"Hausdorff Distance between {anchor_key} and {key}: {value}")
    return value
            
def evaluation_rmsd(anchor_points, target_points, anchor_key, key):
    value = np.sqrt(np.mean(np.linalg.norm(anchor_points - target_points, axis=1)**2))
    print(f"Root Mean Square Deviation between {anchor_key} and {key}: {value}")
    return value
            
def evaluation_chamfer(anchor_points, target_points, anchor_key, key):
    value = np.mean(np.linalg.norm(anchor_points - target_points, axis=1)) + np.mean(np.linalg.norm(anchor_points - target_points, axis=1))
    print(f"Chamfer Distance between {anchor_key} and {key}: {value}")
    return value