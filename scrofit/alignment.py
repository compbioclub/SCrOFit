import numpy as np
#import open3d as o3d
from pycpd import RigidRegistration
#from scipy.spatial import ConvexHull
#from sklearn.decomposition import PCA

def align_slides(adata_dict, anchor_key=None, method='icp', icp_threshold=0.02):
    if method == 'scaling':
        align_slides_by_scaling(adata_dict, anchor_key=anchor_key)
    elif method == 'icp':
        # Iterative Closest Point
        align_slides_by_icp(adata_dict, anchor_key=anchor_key, threshold=icp_threshold)
    elif method == 'rir':
        # Rigid Image Registration
        align_slides_by_rir(adata_dict, anchor_key=anchor_key)
    elif method == 'convex':
        # Convex Hull Centroid alignment
        align_slides_by_convex(adata_dict, anchor_key=anchor_key)
    elif method == 'pca':
        # Principal Component Analysis alignment
        align_slides_by_pca(adata_dict, anchor_key=anchor_key)
    else:
        raise ValueError("Unknown method: {}, please set the method to be 'scaling', 'icp', 'rir', 'convex', and 'pca'.".format(method))

###### helper function #######

def scaling(points):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    return (points - min_vals) / (max_vals - min_vals), min_vals, max_vals

def normalize_points(points):
    return (points - np.mean(points, axis=0)) / np.std(points, axis=0)
        
def numpy_to_open3d(points):
    points_3d = np.hstack((points, np.zeros((points.shape[0], 1))))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    return pcd

def align_icp(source_pcd, target_pcd, threshold, trans_init=np.eye(4)):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

def apply_transformation(pcd, transformation, min_vals, max_vals):
    pcd.transform(transformation)
    transformed_points = np.asarray(pcd.points)[:, :2]
    return transformed_points * (max_vals - min_vals) + min_vals

def compute_convex_hull(data):
    hull = ConvexHull(data)
    return data[hull.vertices]

def compute_centroid(data):
    return np.mean(data, axis=0)

def compute_scale(hull1, hull2):
    distances1 = np.linalg.norm(hull1 - np.roll(hull1, 1, axis=0), axis=1)
    distances2 = np.linalg.norm(hull2 - np.roll(hull2, 1, axis=0), axis=1)
    scale = np.mean(distances1) / np.mean(distances2)
    return scale

###### alignment methods #######

def align_slides_by_scaling(adata_dict, anchor_key=None):
    for key, adata in adata_dict.items():        
        points = adata.obsm['spatial'].copy()
        points_normalized, min_vals, max_vals = scaling(points)
        adata.obsm['spatial_normalized'] = points_normalized
        adata.uns['spatial_scaling_min_vals'] = min_vals
        adata.uns['spatial_scaling_max_vals'] = max_vals

    if anchor_key is not None:
        anchor_min_vals = adata_dict[anchor_key].uns['spatial_scaling_min_vals']
        anchor_max_vals = adata_dict[anchor_key].uns['spatial_scaling_max_vals']
        for key, adata in adata_dict.items():
            if key != anchor_key:
                points = adata.obsm['spatial_normalized'].copy()
                points_scaled = points * (anchor_max_vals - anchor_min_vals) + anchor_min_vals
                adata.obsm['spatial_scaled'] = points_scaled
            else:
                adata.obsm['spatial_scaled'] = adata.obsm['spatial']

def align_slides_by_icp(adata_dict, anchor_key, threshold):
    align_slides_by_scaling(adata_dict)
    anchor_points_normalized = adata_dict[anchor_key].obsm['spatial_normalized']
    anchor_min_vals = adata_dict[anchor_key].uns['spatial_scaling_min_vals']
    anchor_max_vals = adata_dict[anchor_key].uns['spatial_scaling_max_vals']
    
    anchor_pcd = numpy_to_open3d(anchor_points_normalized)
    for key, adata in adata_dict.items():        
        if key == anchor_key:
            adata_dict[anchor_key].obsm['spatial_icp'] = adata_dict[anchor_key].obsm['spatial']
        else:
            points_normalized = adata.obsm['spatial_normalized']
            pcd = numpy_to_open3d(points_normalized)
            transformation = align_icp(pcd, anchor_pcd, threshold)
            aligned_points = apply_transformation(pcd, transformation, anchor_min_vals, anchor_max_vals)
            adata.obsm['spatial_icp'] = aligned_points

def align_slides_by_rir(adata_dict, anchor_key):
    anchor_points = adata_dict[anchor_key].obsm['spatial'].copy()
    anchor_points_normalized = normalize_points(anchor_points)
    adata_dict[anchor_key].obsm['spatial_normalized'] = anchor_points_normalized

    for key, adata in adata_dict.items():        
        if key == anchor_key:
            adata_dict[anchor_key].obsm['spatial_rir'] = anchor_points
        else:
            points_normalized = normalize_points(adata.obsm['spatial'].copy())
            adata.obsm['spatial_normalized'] = points_normalized
            reg = RigidRegistration(X=anchor_points_normalized, Y=points_normalized)
            reg.register()
            aligned_points = reg.TY
            aligned_points = aligned_points * np.std(anchor_points, axis=0) + np.mean(anchor_points, axis=0)          
            adata.obsm['spatial_rir'] = aligned_points
            
def align_slides_by_convex(adata_dict, anchor_key):
    anchor_points = adata_dict[anchor_key].obsm['spatial'].copy()
    hull_anchor = compute_convex_hull(anchor_points)
    centroid_anchor = compute_centroid(hull_anchor)
    hull_centered_anchor = hull_anchor - centroid_anchor
    for key, adata in adata_dict.items():
        if key == anchor_key:
            adata_dict[anchor_key].obsm['spatial_convex'] = anchor_points
        else:
            points2 = adata.obsm['spatial'].copy()
            hull2 = compute_convex_hull(points2)
            centroid2 = compute_centroid(hull2)
            translation = centroid_anchor - centroid2
            points2_translated = points2 + translation
            hull2_centered = hull2 - centroid2
            angle = np.arctan2(hull_centered_anchor[:, 1], hull_centered_anchor[:, 0]).mean() - np.arctan2(hull2_centered[:, 1], hull2_centered[:, 0]).mean()
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            points2_rotated = np.dot(points2_translated - centroid_anchor, rotation_matrix.T) + centroid_anchor
            scale = compute_scale(hull_anchor, hull2)
            aligned_points = (points2_rotated - centroid_anchor) * scale + centroid_anchor
            adata.obsm['spatial_convex'] = aligned_points
            
def align_slides_by_pca(adata_dict, anchor_key):
    align_slides_by_scaling(adata_dict)
    anchor_min_vals = adata_dict[anchor_key].uns['spatial_scaling_min_vals']
    anchor_max_vals = adata_dict[anchor_key].uns['spatial_scaling_max_vals']
    anchor_points = adata_dict[anchor_key].obsm['spatial_normalized'].copy()
    pca_anchor = PCA(n_components=2)
    anchor_transformed = pca_anchor.fit_transform(anchor_points)
    
    for key, adata in adata_dict.items():
        if key == anchor_key:
            adata_dict[anchor_key].obsm['spatial_pca'] = anchor_transformed * (anchor_max_vals - anchor_min_vals) + anchor_min_vals
        else:
            points2 = adata.obsm['spatial_normalized'].copy()
            pca_points2 = PCA(n_components=2)
            points2_transformed = pca_points2.fit_transform(points2)
            aligned_points = points2_transformed * (anchor_max_vals - anchor_min_vals) + anchor_min_vals
            adata.obsm['spatial_pca'] = aligned_points