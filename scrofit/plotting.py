import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, to_rgba
from scipy import stats
import scanpy as sc
import squidpy as sq


from scrofit.util import get_array


def get_value_boundary(v):
    if type(v) == pd.DataFrame:
        v = v.to_numpy()
    if abs(v.max()) > abs(v.min()):
        return abs(v.max())
    else:
        return abs(v.min())

def plt_util(title):
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.colorbar()
    


def customize_cmap(cmap, position='center', color='black'):
    if cmap is None:
        base_cmap = customize_exp_cmap()
    else:
        base_cmap = plt.get_cmap(cmap)
    # Define a custom colormap with the value 0 set to black
    colors = [base_cmap(i) for i in range(base_cmap.N)]

    if position == 'center':
        colors[base_cmap.N // 2] = to_rgba(color)  # Set the middle color
    if position == 'min':
        colors[0] = to_rgba(color)
    if position == 'max':
        colors[-1] = to_rgba(color)
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=base_cmap.N)
    return custom_cmap


def customize_exp_cmap():
    cmap_name = 'viridis_r'
    base_cmap = plt.get_cmap(cmap_name)

    truncate_ratio = 0.7
    colors = base_cmap(np.linspace(0, truncate_ratio, 256))[::-1]
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    return custom_cmap


def plot_CCS(obj, s=1, ncol=3, nrow=None, figsize=(10, 10), fig_fn=None):

    adata_dict = obj.adata_dict
    adata = list(adata_dict.values())[1]

    fig = plt.figure(figsize=figsize)
    nsub = len(adata.obsm.keys()) * len(adata_dict)
    if nrow is None:
        nrow = int(np.ceil(nsub / ncol))

    i = 1
    for key, adata in adata_dict.items():
        plt.subplot(nrow, ncol, i)
        points = adata.obsm['spatial']
        plt.scatter(points[:, 0], points[:, 1], s=s)
        plt.title(f'OCS: {key}\n Size: {points.shape[0]}')
        i += 1

    for ccs_type in adata.obsm.keys():
        if not ccs_type.startswith('spatial_'):
            continue
        if ccs_type.endswith('_map'):
            continue        
        plt.subplot(nrow, ncol, i)
        for key, adata in adata_dict.items():
            if ccs_type in adata.obsm:
                points = adata.obsm[ccs_type]
                plt.scatter(points[:, 0], points[:, 1], label=key, s=s)
                plt.legend()
        plt.title(f'CCS: {ccs_type}')
        i += 1

    for ccs_type in adata.obsm.keys():
        if not ccs_type.startswith('spatial_'):
            continue
        if not ccs_type.endswith('_map'):
            continue        
        plt.subplot(nrow, ncol, i)
        for key, adata in adata_dict.items():
            if ccs_type in adata.obsm:
                points = adata.obsm[ccs_type]
                plt.scatter(points[:, 0], points[:, 1], label=key, s=s)
                plt.legend()
        plt.title(f'OCS: {ccs_type}\n Size: {points.shape[0]}')
        i += 1

    #plt.tight_layout()
    plt.suptitle(obj.sample)
    if fig_fn is not None:
        fig.savefig(fig_fn)


def _spatial_scatter(adata, color, ccs_type='spatial',
                    s=5, cmap='viridis'):

    coords = adata.obsm[ccs_type]
    plt.scatter(coords[:,0], coords[:,1], c=adata[:,color].X.toarray(), s=s, cmap=cmap)
    plt_util(color)

def embed(adata, color=None, figsize=(2,2), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.umap(adata, color=color, size=15, 
               na_color='black', ax=ax, show=False,
               legend_loc=None,
               **kwargs
               )
    ax.set_xlabel("SCrOFit Embed1")
    ax.set_ylabel("SCrOFit Embed2")
    fig.tight_layout()
    plt.show()    

def ocs_spatial_scatter(adata, key, figsize=None,
                        **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    if f'{key}_spatial' in adata.obsm.keys():
        spatial_key = f'{key}_spatial'
    else:
        spatial_key = 'spatial'
    sq.pl.spatial_scatter(
        adata, spatial_key=spatial_key, ax=ax,
        na_color="black",  # Set NA color to black
        legend_na=True,    # Explicitly include NA in legend
        **kwargs)
    ax.set_xlabel(f"{key} OCS1")
    ax.set_ylabel(f"{key} OCS2")
    fig.tight_layout()
    plt.show()  

def ccs_spatial_scatter(adata, key, figsize=None,
                        **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    spatial_key = f'{key}_spatial_rir'
    sq.pl.spatial_scatter(
        adata, spatial_key=spatial_key, ax=ax,
        na_color="black",  # Set NA color to black
        legend_na=True,    # Explicitly include NA in legend
        **kwargs)
    ax.set_xlabel("SCrOFit CCS1")
    ax.set_ylabel("SCrOFit CCS2")
    fig.tight_layout()
    plt.show()       

from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def ccs_spatial_zoom(
    st_adata, sm_adata,
    metabolite=None, 
    gene=None,
    x_min=5000,
    x_max = 10000, 
    y_min = 5000,
    y_max = 10000,
    cell_type_col="ST_Cell_Type",
    st_coord_key="ST_spatial_rir",
    sm_coord_key="SM_spatial_rir",
    gene_source="var_names",   # <- use var_names for molecule gene labels
    cell_dot_size=200,
    gene_dot_size=10,
    figsize=(7, 7),
    invert_y=True,
    st_cmap='Accent',
    g_cmap='Spectral_r',
    m_cmap='plasma'
):

    # --- Spot-level coordinates and colours ---------------------------------
    st_coords = pd.DataFrame(
        st_adata.obsm[st_coord_key],
        columns=["x", "y"],
        index=st_adata.obs_names,
    ).assign(cell_type=st_adata.obs[cell_type_col].astype("category"))
    if gene is not None:
        st_coords[gene] = st_adata[:, gene].to_df().to_numpy()


    cell_types = st_coords["cell_type"].cat.categories

    if st_cmap:
        base_colors = plt.get_cmap(st_cmap).colors  # length 8
        if st_cmap == 'Accent':
            base_colors = base_colors[:3] + base_colors[-4:]
    else:
        base_colors = sc.pl.palettes.default_20 

    base_colors = base_colors[:len(cell_types)]
    type_colors = dict(zip(cell_types, base_colors))

    # --- Molecule-level coordinates -----------------------------------------
    sm_coords = pd.DataFrame(
        sm_adata.obsm[sm_coord_key],
        columns=["x", "y"],
    )
    if metabolite is not None:
        sm_coords[metabolite] = sm_adata[:,metabolite].to_df().to_numpy()

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    st_coords = st_coords[
        (st_coords['x'] > x_min) &
        (st_coords['x'] < x_max) &
        (st_coords['y'] > y_min) &
        (st_coords['y'] < y_max)]
    sm_coords = sm_coords[
        (sm_coords['x'] > x_min) &
        (sm_coords['x'] < x_max) &
        (sm_coords['y'] > y_min) &
        (sm_coords['y'] < y_max)]

    for ct, subset in st_coords.groupby("cell_type"):
        ax.scatter(
            subset["x"],
            subset["y"],
            s=cell_dot_size,
            marker="o",
            linewidths=1,
            facecolors='none', 
            edgecolors=type_colors[ct],
            alpha=1,
            label=str(ct),
        )

    if metabolite:
        ax.scatter(
            sm_coords["x"],
            sm_coords["y"],
            s=gene_dot_size,
            c=sm_coords[metabolite].astype(float),
            cmap=m_cmap,
            alpha=1,
            label=f"{metabolite}",
        )
    if gene:
        ax.scatter(
            st_coords["x"],
            st_coords["y"],
            s=gene_dot_size,
            marker='x',
            c=st_coords[gene].astype(float),
            cmap=g_cmap,
            alpha=1,
            label=f"{gene}",
        )


    if invert_y:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])        
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from plotnine import *


def violin(adata, compare_type, dopamine, comparisons,
             fig_size=(5, 5), 
             palette=sc.pl.palettes.default_20,
             text_x_angle=0):

    df = adata.obs.sort_values(compare_type).copy()
    df['dopamine'] = df[dopamine].astype(float)
    df = df[~df[compare_type].isna()]

    cat_order = df[compare_type].unique().tolist()
    group_max = df.groupby(compare_type)['dopamine'].max()
    global_max = group_max.max()

    annot_info = []
    offset = global_max * 0.1 
    for i, (g1, g2) in enumerate(comparisons):
        x1, x2 = cat_order.index(g1) + 1, cat_order.index(g2) + 1
        y = global_max + offset * (i+1)   
        grp1 = df.loc[df[compare_type]==g1, 'dopamine']
        grp2 = df.loc[df[compare_type]==g2, 'dopamine']
        stat, pval = mannwhitneyu(grp1, grp2)
        annot_info.append({
            'x1': x1, 'x2': x2,
            'y': y, 'pval': pval,
            'x_text': (x1 + x2) / 2,
            'y_text': y + offset*0.5
        })

    if type(palette) is list:
        myfill = {cat: palette[i % len(palette)] for i, cat in enumerate(cat_order)}
    else:
        myfill = palette

    p = (
        ggplot(df, aes(x=compare_type, y='dopamine', fill=compare_type))
        + geom_violin(trim=True, width=0.8)
        + geom_boxplot(
            width=0.03,
            fill="white",       # white fill for contrast
            color="black",      # black outline
            outlier_size=0.5
        )        
        + scale_fill_manual(values=myfill)
        + theme_bw()
        + theme(
            axis_text_x=element_text(angle=text_x_angle, hjust=1),
            legend_position='none',
            panel_grid = element_blank(),
            figure_size = fig_size
        )
        + labs(x='', y=dopamine)
    )

    for ann in annot_info:
        p = p + annotate(
            'segment',
            x=ann['x1'], xend=ann['x2'],
            y=ann['y'], yend=ann['y'],
            size=0.5, color='black'
        ) + annotate(
            'text',
            x=ann['x_text'], y=ann['y_text'],
            label=f"p = {ann['pval']:.2e}",
            ha='center', size=8, color='black'
        )

    return p    

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from plotnine import (
    ggplot, aes, geom_point,
    annotate, labs, theme_bw, 
    theme, element_text
)
from anndata import AnnData

def plot_gene_cell_type(st_adata, mdata, gene, compare_header, comapre_type):
    adata = st_adata.copy()
    sample='V11T17-102'
    adata.obs[compare_header][adata.obs[compare_header] != comapre_type] = np.nan
    ocs_spatial_scatter(adata, 'ST', color=compare_header, img=True, shape=None, 
                          library_id=sample, cmap='Spectral_r', size=2) 
    adata[adata[adata.obs[compare_header] != comapre_type].obs_names, gene].X = np.nan
    ocs_spatial_scatter(adata, 'ST', color=gene, img=True, shape=None, 
                          library_id=sample, cmap='Spectral_r', size=2)  
    adata = mdata.copy()
    adata.obs = adata.obs.reset_index()
    adata.obs[f'ST_{compare_header}'][adata.obs[f'ST_{compare_header}'] != comapre_type] = np.nan
    adata[adata[adata.obs[f'ST_{compare_header}'] != comapre_type].obs_names, gene].X = np.nan
    adata[adata[adata.obs[f'ST_{compare_header}'] != comapre_type].obs_names, 'mz_674.2805'].X = np.nan
    ccs_spatial_scatter(adata, 'SM', color='mz_674.2805', img=True, shape=None, 
                          library_id=sample, cmap='plasma', size=2)
    embed(adata, color=f'ST_{compare_header}', figsize=(1.8, 2))
    embed(adata, color=gene, cmap='Spectral_r', figsize=(2, 2))
    embed(adata, color='mz_674.2805', cmap='plasma', figsize=(2.1, 2))

    p = scatter_two_genes(adata, gene, 'mz_674.2805')
    return p


def scatter_two_genes(
    adata: AnnData, 
    gene_x: str, 
    gene_y: str,
    figsize = (1.8,2)
):
    """
    Plot a scatter of gene_x vs gene_y expression in adata.obs
    and annotate the Pearson correlation.
    """
    # 1) extract expression vectors
    #    adata[:, gene].X may be sparse or dense, so we flatten safely:
    def get_vector(gene):
        v = adata[:, gene].X
        if hasattr(v, "toarray"):
            v = v.toarray()
        return np.ravel(v)
    
    x = get_vector(gene_x)
    y = get_vector(gene_y)
        
    # 2) build DataFrame
    df = pd.DataFrame({gene_x: x, gene_y: y})
    df = df.dropna(subset=[gene_x, gene_y]).copy()

    # 3) compute Pearson r
    r, pval = pearsonr(df[gene_x], df[gene_y])
    title = f"rho = {r:.2f}\np = {pval:.1e}"
    
    # 5) plot with plotnine
    p = (
        ggplot(df, aes(x=gene_x, y=gene_y))
        + geom_point(alpha=0.5, color='lightgray')
        + labs(
            title=title,
            x=gene_x,
            y=gene_y
        )
        + geom_smooth(method='lm', se=True, color='red', size=0.7, linetype='dashed')
        + theme_bw()
        + theme(
            plot_title = element_text(size=12, weight="bold"),
            axis_title = element_text(size=10),
            axis_text  = element_text(size=8),
            figure_size = figsize,
            panel_grid=element_blank(),
        )
    )
    
    return p

