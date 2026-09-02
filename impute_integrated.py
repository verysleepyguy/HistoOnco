import argparse
import multiprocessing
import os

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import numpy as np
from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import (
        read_lines, read_string, save_pickle, load_image, load_tsv,
        write_lines)
from image import get_disk_mask
from train import get_model as train_load_model
# from reduce_dim import reduce_dim
from visual import plot_matrix, plot_spot_masked_image
import pandas as pd
import math
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# =========================== HistoOnco PARAMETERS ============================
# HistoOnco repurposes iSCALE to predict cancer-pathway activity scores (UCell on
# MSigDB Hallmark + curated C6 oncogenic sets) from H&E histology features, instead
# of iSCALE's original target of raw per-superpixel gene counts.
#
# The knobs below are the only things you should normally need to touch. They are
# collected here (not buried inside the loss / data code) on purpose, matching the
# style of the other top-of-pipeline constants such as ``factor = 16``.

# True  -> HistoOnco mode: one label row per CELL (10x Atera single-cell-resolution
#          spatial data), direct per-cell supervision, plus a spatial-smoothness
#          regularizer. The Visium spot-pooling ("forward-sum") machinery and the
#          multi-daughter-capture stitching are bypassed (see notes at each site).
# False -> original iSCALE behavior, fully preserved for Visium spot-level and
#          multi-capture datasets. Nothing on the False path was modified.
SINGLE_CELL_MODE = True

# Where the pathway-score label table lives. None -> auto-detect, trying (in order)
#   {prefix}ucell_pathway_scores.parquet
#   {prefix}ucell_pathway_scores.csv.gz
#   {prefix}ucell_pathway_scores.csv
# Set an explicit path (any of .parquet / .csv.gz / .csv) to override.
LABEL_FILE = None

# Pathway-score columns are selected generically by this prefix. The COUNT and the
# NAMES of pathways are never hardcoded anywhere: n_out is derived from how many
# columns match, exactly as iSCALE derives n_out from gene-names.txt / cnts.tsv.
PATHWAY_COL_PREFIX = 'UCell_'

# ---- spatial-smoothness regularizer:  L_total = L_pred + SPATIAL_LOSS_WEIGHT * L_spatial
# L_pred    : plain MSE between each cell's predicted pathway-score vector and its
#             own real UCell label vector.
# L_spatial : MSE between a cell's prediction and the mean prediction over its
#             SPATIAL_KNN_K nearest spatial neighbors (plain Euclidean distance on
#             the x/y columns; NOT grouped or restricted by cell type - this is a
#             deliberate simplification vs. PASTA's cell-type-conditioned loss).
SPATIAL_KNN_K = 6
SPATIAL_LOSS_WEIGHT = 0.1

# Cells whose (rescaled) pixel location lands within this many pixels of the
# embedding-grid border are dropped, so the single-pixel feature lookup is valid.
LABEL_BORDER_MARGIN = 1
# ===========================================================================

class FeedForward(nn.Module):

    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            # TODO: change activation to LeakyRelu(0.01)
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x, indices=None):
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class ForwardSumModel(pl.LightningModule):

    def __init__(self, lr, n_inp, n_out):
        super().__init__()
        self.lr = lr
        self.net_lat = nn.Sequential(
                FeedForward(n_inp, 256),
                FeedForward(256, 256),
                FeedForward(256, 256),
                FeedForward(256, 256))
        self.net_out = FeedForward(
                256, n_out,
                activation=ELU(alpha=0.01, beta=0.01))
        self.save_hyperparameters()

    def inp_to_lat(self, x):
        return self.net_lat.forward(x)

    def lat_to_out(self, x, indices=None):
        x = self.net_out.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.inp_to_lat(x)
        x = self.lat_to_out(x, indices)
        return x

    def training_step(self, batch, batch_idx):
        x, y_mean = batch
        y_pred = self.forward(x)
        y_mean_pred = y_pred.mean(-2)
        # TODO: try l1 loss
        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        self.log('rmse', mse**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class SpatialSmoothModel(ForwardSumModel):
    """HistoOnco prediction head + loss (used only when SINGLE_CELL_MODE=True).

    Same network as ``ForwardSumModel`` (4x256 LeakyReLU(0.1) latent stack + ELU
    output head, predicting from frozen HIPT/ViT histology features - unchanged).
    Only the training objective differs:

        L_total = L_pred + lambda * L_spatial

    * L_pred    - ordinary MSE between each cell's predicted pathway-score vector
                  and that same cell's real UCell label vector. This REPLACES
                  iSCALE's spot-pooled MSE (``y_pred.mean(-2)`` vs a Visium spot
                  average): our labels are already at single-cell resolution, so
                  there is no disk of superpixels to pool over.

    * L_spatial - MSE between a cell's prediction and the mean prediction over its
                  k nearest SPATIAL neighbors (Euclidean distance on x/y, computed
                  once up front - see CellDataset). We use squared error rather
                  than cosine similarity so that L_spatial and L_pred share units
                  and ``lambda`` is directly interpretable as a relative weight;
                  cosine would ignore the magnitude of the pathway-activity
                  difference, which is exactly the quantity we want to be smooth.
                  The neighbor term is on PREDICTIONS (the neighbor feature vectors
                  are forwarded in the same step), not on labels, so it acts as a
                  genuine smoothness prior on the model output.

    ``k`` and ``lambda`` come from the top-of-file PARAMETERS block via kwargs.
    """

    def __init__(self, lr, n_inp, n_out,
                 spatial_weight=SPATIAL_LOSS_WEIGHT, spatial_k=SPATIAL_KNN_K):
        super().__init__(lr=lr, n_inp=n_inp, n_out=n_out)
        self.spatial_weight = spatial_weight
        self.spatial_k = spatial_k
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # x_self: (B, C)   x_nbr: (B, k, C)   y: (B, n_out)
        x_self, x_nbr, y = batch

        p_self = self.forward(x_self)                     # (B, n_out)
        l_pred = ((p_self - y) ** 2).mean()

        b, k, c = x_nbr.shape
        p_nbr = self.forward(x_nbr.reshape(b * k, c)).reshape(b, k, -1)
        p_nbr_mean = p_nbr.mean(1)                        # (B, n_out)
        l_spatial = ((p_self - p_nbr_mean) ** 2).mean()

        loss = l_pred + self.spatial_weight * l_spatial
        self.log('rmse', l_pred ** 0.5, prog_bar=True)
        self.log('l_spatial', l_spatial, prog_bar=True)
        return loss


class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius):
        super().__init__()
        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)
        isin = np.isfinite(x).all((-1, -2))
        self.x = x[isin]
        self.y = y[isin]
        self.locs = locs[isin]
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def show(self, channel_x, channel_y, prefix):
        mask = self.mask
        size = self.size
        locs = self.locs
        xs = self.x
        ys = self.y

        plot_spot_masked_image(
                locs=locs, values=xs[:, :, channel_x], mask=mask, size=size,
                outfile=f'{prefix}x{channel_x:04d}.png')

        plot_spot_masked_image(
                locs=locs, values=ys[:, channel_y], mask=mask, size=size,
                outfile=f'{prefix}y{channel_y:04d}.png')


class CellDataset(Dataset):
    """Direct per-cell supervision (used only when SINGLE_CELL_MODE=True).

    Deliberately does NOT use ``get_disk_mask`` / ``get_patches_flat``: those exist
    in ``SpotDataset`` to pool many superpixel feature vectors under a Visium
    spot's disk, because iSCALE's ground truth only existed at spot resolution.
    Our label file (10x Atera) has one row per cell at true single-cell
    resolution, so each cell is supervised directly against the single frozen
    histology-feature vector at its own pixel location. ``SpotDataset`` is left
    untouched for the original spot-level / multi-capture use case.

    __getitem__ returns a triple consumed by ``SpatialSmoothModel.training_step``:
        x_self : (C,)      feature vector at this cell's pixel
        x_nbr  : (k, C)    feature vectors at this cell's k nearest spatial
                           neighbors (indices precomputed once, in get_data_cells)
        y_self : (n_out,)  this cell's own UCell pathway-score row
    """

    def __init__(self, x_all, y, locs, nbr_idx):
        super().__init__()
        # one feature vector per cell (single-pixel lookup, no disk mask)
        x = x_all[locs[:, 0], locs[:, 1]]
        isin = np.isfinite(x).all(-1)
        if not isin.all():
            # keep cell / neighbor-index arrays consistent after dropping cells
            keep = np.flatnonzero(isin)
            remap = np.full(len(isin), -1, dtype=np.int64)
            remap[keep] = np.arange(keep.size)
            nbr_idx = remap[nbr_idx[keep]]
            # any neighbor that was itself dropped -> point at self (no-op in loss)
            self_col = np.arange(keep.size)[:, None]
            nbr_idx = np.where(nbr_idx < 0, self_col, nbr_idx)
            x = x[keep]
            y = y[keep]
            locs = locs[keep]

        self.x = x.astype(np.float32)
        self.y = np.asarray(y).astype(np.float32)
        self.locs = locs
        self.nbr_idx = nbr_idx.astype(np.int64)
        self.size = x_all.shape[:2]
        # SpotDataset exposes a `.mask`; downstream code checks it. There is no
        # pooling mask here, so expose None and let impute() special-case it.
        self.mask = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.x[self.nbr_idx[idx]], self.y[idx]

    def show(self, *args, **kwargs):
        # training-data plots in SpotDataset visualize the disk patches; nothing
        # analogous for single-pixel per-cell features, so this is intentionally
        # a no-op (keeps get_model's dataset.show(...) call site unchanged).
        pass


def get_disk(img, ij, radius):
    i, j = ij
    patch = img[i-radius:i+radius, j-radius:j+radius]
    disk_mask = get_disk_mask(radius)
    patch[~disk_mask] = 0.0
    return patch


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]

        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list


def add_coords(embs):
    coords = np.stack(np.meshgrid(
            np.linspace(-1, 1, embs.shape[0]),
            np.linspace(-1, 1, embs.shape[1]),
            indexing='ij'), -1)
    coords = coords.astype(embs.dtype)
    mask = np.isfinite(embs).all(-1)
    coords[~mask] = np.nan
    embs = np.concatenate([embs, coords], -1)
    return embs


# def reduce_embeddings(embs):
#     # cls features
#     cls, __ = reduce_dim(embs[..., :192], 0.99)
#     # sub features
#     sub, __ = reduce_dim(embs[..., 192:-3], 0.90)
#     rgb = embs[..., -3:]
#     embs = np.concatenate([cls, sub, rgb], -1)
#     return embs


def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    embs = get_embeddings(prefix)
    # embs = embs[..., :192]  # use high-level features only
    # embs = reduce_embeddings(embs)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    # embs = add_coords(embs)
    return embs, cnts, locs


def get_data_smooth(prefix, radius, dist):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    embs = get_embeddings(prefix)

    #normalize data to remove batch effects across daughter captures prior to smoothing
    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)

    # Extract coordinates as numpy array
    adata = pd.read_csv(f'{prefix}locs.tsv', sep = '\t')
    coords = np.array(adata.iloc[:,[1,2]])
    
    # Compute pairwise distances
    distance_matrix = squareform(pdist(coords, metric='euclidean'))

    # Perform clustering with a maximum distance of 100
    db = DBSCAN(eps=dist, min_samples=1, metric='precomputed')
    clusters = db.fit_predict(distance_matrix)
    locs = adata.iloc[:,[1,2]]

    # Add cluster labels to the locs DataFrame
    locs['cluster'] = clusters

    # Compute average locations for each cluster
    locs2 = locs.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).reset_index(drop=True)

    # Compute average gene expressions for each cluster
    cnts['cluster'] = clusters
    cnts2 = cnts.groupby('cluster').mean().reset_index(drop=True)

    print("Original matrix sizes (locs, cnts):")
    print(locs.shape)
    print(cnts.shape)

    print("Integrated matrix sizes (locs, cnts):")
    print(locs2.shape)
    print(cnts2.shape)

    print("Embeddings shape")
    print(embs.shape)

  # change xy coordinates to ij coordinates
    locs2 = np.stack([locs2['y'], locs2['x']], -1)

    target_shape=embs.shape[:2]

    # match coordinates of embeddings and spot locations
    if target_shape is not None:
        wsi = load_image(f'{prefix}he.tiff')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs2 = locs2.astype(float)
        locs2 /= rescale_factor

    # find the nearest pixel
    locs2 = locs2.round().astype(int)

    print("min x", min(locs2[:,0]))
    print("max x", max(locs2[:,0]))

    print("min y", min(locs2[:,1]))
    print("max y", max(locs2[:,1]))

    print(math.ceil(radius))

    # embs = embs[..., :192]  # use high-level features only
    # embs = reduce_embeddings(embs)
    #locs = get_locs(prefix, target_shape=embs.shape[:2])
    # embs = add_coords(embs)

    #which locs are outside of the histology image? (we want to remove these spots)
    x_out_low = locs2[:,0] < math.ceil(radius)  
    x_out_high = locs2[:,0] > embs.shape[0]-math.ceil(radius) 

    y_out_low = locs2[:,1] < math.ceil(radius) 
    y_out_high = locs2[:,1] > embs.shape[1]-math.ceil(radius) 

    remove = x_out_high
    for i in range(0,len(x_out_high)):
      remove[i] = x_out_low[i] or x_out_high[i] or y_out_low[i] or y_out_high[i]

    keep = ~remove
    locs2 = locs2[keep]
    cnts2 = cnts2[keep]

    print("Integrated matrix sizes (locs, cnts) after filtering border spots:")
    print(locs2.shape)
    print(cnts2.shape)

    return embs, cnts2, locs2


# ===================== HistoOnco single-cell data path ======================
# Everything below is only reached when SINGLE_CELL_MODE=True. get_data_smooth
# above (DBSCAN merge of overlapping daughter captures + cross-capture min-max
# normalization) is left completely intact for the multi-capture Visium case.


def _resolve_label_file(prefix):
    """Return the path to the pathway-score label table.

    LABEL_FILE (top of file) wins if set. Otherwise auto-detect by extension in
    priority order parquet > csv.gz > csv.
    """
    if LABEL_FILE is not None:
        if not os.path.exists(LABEL_FILE):
            raise FileNotFoundError(
                    f'LABEL_FILE was set to {LABEL_FILE!r} but that file does '
                    f'not exist.')
        return LABEL_FILE
    candidates = [
            f'{prefix}ucell_pathway_scores.parquet',
            f'{prefix}ucell_pathway_scores.csv.gz',
            f'{prefix}ucell_pathway_scores.csv']
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
            'Could not find a pathway-score label file. Looked for '
            + ', '.join(repr(c) for c in candidates)
            + '. Set LABEL_FILE at the top of impute_integrated.py to an '
              'explicit path (.parquet, .csv.gz or .csv).')


def get_pathway_labels(path):
    """Load the UCell pathway-score table produced by ucell_spatial_pipeline.ipynb.

    Accepts .parquet, .csv.gz or .csv (auto-detected from the extension). The
    table has columns ``cell_id, x, y`` then one ``UCell_<pathway>`` column per
    retained pathway.

    Returns
    -------
    xy         : (N, 2) float array of raw x/y cell-centroid coordinates
    labels     : (N, P) float32 DataFrame, columns = the matched pathway columns
                 (order preserved). P (= n_out) is whatever the file contains -
                 never hardcoded.
    """
    lower = path.lower()
    if lower.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif lower.endswith('.csv.gz') or lower.endswith('.csv'):
        df = pd.read_csv(path)  # pandas handles .gz transparently
    else:
        raise ValueError(
                f'Unsupported label-file extension: {path!r}. '
                f'Expected .parquet, .csv.gz or .csv.')
    print(f'Pathway-score table loaded from {path}  (shape {df.shape})')

    # select pathway columns generically by prefix - count and names are derived
    # from the file, exactly as iSCALE derives n_out from gene-names.txt.
    pathway_cols = [c for c in df.columns if str(c).startswith(PATHWAY_COL_PREFIX)]
    if len(pathway_cols) == 0:
        raise ValueError(
                f'No {PATHWAY_COL_PREFIX!r}-prefixed pathway columns found in '
                f'{path!r}. Columns present: {list(df.columns)}. Refusing to '
                f'proceed with a zero-width output layer.')

    for col in ('x', 'y'):
        if col not in df.columns:
            raise ValueError(
                    f'Label file {path!r} is missing the {col!r} column '
                    f'(needed for spatial-neighbor computation and for '
                    f'registering cells to the histology image).')

    print(f'  {len(pathway_cols)} pathway columns matched '
          f'{PATHWAY_COL_PREFIX!r}: {pathway_cols[:3]}'
          + (' ...' if len(pathway_cols) > 3 else ''))

    xy = df[['x', 'y']].to_numpy().astype(float)
    labels = df[pathway_cols].astype(np.float32).reset_index(drop=True)
    return xy, labels


def get_data_cells(prefix):
    """HistoOnco replacement for get_data_smooth on single-cell-resolution data.

    Differences from get_data_smooth, and why:

    * NO DBSCAN cluster-merge / no per-cluster averaging. That step stitches
      spots from overlapping daughter Visium captures onto one mother image.
      Our dataset is a single whole-slide Atera capture, and merging nearby
      rows here would fuse genuinely distinct cells and destroy the single-cell
      resolution that is the whole point. One row in -> one cell out.

    * NO cross-capture min-max normalization of the target. UCell scores are
      rank-based, already in [0, 1], and comparable across pathways; with a
      single capture there is no inter-capture batch effect to correct. L_pred
      is therefore plain MSE in real UCell-score units.

    * Targets come from the UCell pathway-score file, not cnts.tsv / select_genes.

    * The frozen histology-feature pipeline (embeddings-hist.pickle) is used
      exactly as before - untouched.

    Also precomputes, once, the k-nearest-spatial-neighbor index used by the
    spatial-smoothness loss, and writes the retained pathway names to
    {prefix}gene-names.txt so the downstream scripts (refine_gene.py,
    evaluate_fit.py, plot_imputed_iSCALE.py) that iterate that file keep working
    unmodified.
    """
    embs = get_embeddings(prefix)

    label_path = _resolve_label_file(prefix)
    xy, labels = get_pathway_labels(label_path)
    names = list(labels.columns)

    # --- register cell centroids to the embedding grid -------------------------
    # Same xy->ij + rescale logic as get_data_smooth / impute_by_basic.get_locs:
    # label x/y are in full-res H&E pixels; embeddings are downsampled.
    locs = np.stack([xy[:, 1], xy[:, 0]], -1)  # (x, y) -> (i=row=y, j=col=x)
    target_shape = np.array(embs.shape[:2])
    wsi = load_image(f'{prefix}he.tiff')
    current_shape = np.array(wsi.shape[:2])
    rescale_factor = current_shape // target_shape
    locs = locs.astype(float) / rescale_factor
    locs = locs.round().astype(int)

    # --- drop cells outside / on the border of the embedding grid -------------
    m = LABEL_BORDER_MARGIN
    inside = (
            (locs[:, 0] >= m) & (locs[:, 0] < embs.shape[0] - m) &
            (locs[:, 1] >= m) & (locs[:, 1] < embs.shape[1] - m))
    n_drop = int((~inside).sum())
    if n_drop:
        print(f'Dropping {n_drop} / {len(inside)} cells outside the '
              f'histology-image border (margin={m}px).')
    xy = xy[inside]
    locs = locs[inside]
    labels = labels.loc[inside].reset_index(drop=True)

    if len(labels) == 0:
        raise ValueError(
                'No cells left after border filtering - check that the label '
                'x/y coordinates are in the same full-res pixel space as '
                f'{prefix}he.tiff.')

    # --- precompute k nearest spatial neighbors ONCE --------------------------
    # Plain Euclidean distance on the raw x/y columns. NOT grouped by cell type
    # (deliberate simplification vs. PASTA). Ask for k+1 because the first
    # neighbor NearestNeighbors returns is the point itself; drop that column.
    k = SPATIAL_KNN_K
    k_eff = min(k, len(labels) - 1)
    if k_eff < k:
        print(f'Only {len(labels)} cells; reducing spatial-neighbor k '
              f'from {k} to {k_eff}.')
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(xy)
    nbr_idx = nn.kneighbors(xy, return_distance=False)[:, 1:]  # (N, k_eff)

    print(f'get_data_cells: {len(labels)} cells, {len(names)} pathways '
          f'(n_out={len(names)}), spatial k={k_eff}')
    print(f'embeddings shape {embs.shape}')

    # keep the downstream iteration list in sync with our targets
    write_lines(names, f'{prefix}gene-names.txt')

    return embs, labels, locs, nbr_idx


def get_model_kwargs(kwargs):
    return get_model(**kwargs)


def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, lr,
        load_saved=True, device='cuda', nbr_idx=None):

    print('x:', x.shape, ', y:', y.shape)

    x = x.copy()

    if nbr_idx is not None:
        # HistoOnco single-cell path: no disk-mask pooling, spatial-smoothness loss
        dataset = CellDataset(x, y, locs, nbr_idx)
        model_class = SpatialSmoothModel
        model_kwargs = dict(
                n_inp=x.shape[-1],
                n_out=y.shape[-1],
                lr=lr,
                spatial_weight=SPATIAL_LOSS_WEIGHT,
                spatial_k=SPATIAL_KNN_K)
    else:
        dataset = SpotDataset(x, y, locs, radius)
        model_class = ForwardSumModel
        model_kwargs = dict(
                n_inp=x.shape[-1],
                n_out=y.shape[-1],
                lr=lr)

    dataset.show(
            channel_x=0, channel_y=0,
            prefix=f'{prefix}iSCALE_output/training-data-plots/')
    model = train_load_model(
            model_class=model_class,
            model_kwargs=model_kwargs,
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset


def normalize(embs, cnts):

    embs = embs.copy()
    cnts = cnts.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)


def show_results(x, names, prefix):
    for name in ['CD19', 'MS4A1', 'ERBB2', 'GNAS']:
        if name in names:
            idx = np.where(names == name)[0][0]
            plot_matrix(x[..., idx], prefix+name+'.png')


def predict_single_out(model, z, indices, names, y_range):
    z = torch.tensor(z, device=model.device)
    y = model.lat_to_out(z, indices=indices)
    y = y.cpu().detach().numpy()
    # y[y < 0.01] = 0.0
    # y[y > 1.0] = 1.0
    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict_single_lat(model, x):
    x = torch.tensor(x, device=model.device)
    z = model.inp_to_lat(x)
    z = z.cpu().detach().numpy()
    return z


# def cluster_lat(x, prefix, device='cuda'):
#     x_minor = x
#     x_major = smoothen(
#             x_minor, size=8, method='cnn', mode='mean',
#             device=device)
#     labels = cluster_hierarchical(
#             x_major.transpose(2, 0, 1), x_minor.transpose(2, 0, 1),
#             method='km', n_clusters=10)
#     # x = reduce_dim(x, method='pca', n_components=0.95)[0]
#     # labels_raw = cluster(
#     #         x.transpose(2, 0, 1), method='km', n_clusters=10)[0]
#     # labels_cls = relabel_small_connected(labels_raw, min_size=1000)
#     # labels_con = cluster_connected(labels_cls)
#     # labels = np.stack([labels_cls, labels_con], -1)
#     plot_labels(labels[..., :2], prefix+'clusters-genes.png')
#     save_pickle(labels, prefix+'clusters-genes.pickle')
#     return labels


def predict(
        model_states, x_batches, name_list, y_range, prefix,
        device='cuda'):

    # states: different initial values for training
    # batches: subsets of observations
    # groups: subsets outcomes

    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]

    # get features of second last layer
    z_states_batches = [
            [predict_single_lat(mod, x_bat) for mod in model_states]
            for x_bat in x_batches]
    z_point = np.concatenate([
        np.median(z_states, 0)
        for z_states in z_states_batches])
    z_dict = dict(cls=z_point.transpose(2, 0, 1))
    save_pickle(
            z_dict,
            prefix+'embeddings-gene.pickle')
    del z_point

    # predict and save y by batches in outcome dimension
    idx_list = np.arange(len(name_list))
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    idx_groups = np.array_split(idx_list, n_groups_outcome)
    for idx_grp in idx_groups:
        name_grp = name_list[idx_grp]
        y_ran = y_range[idx_grp]
        y_grp = np.concatenate([
            np.median([
                predict_single_out(mod, z, idx_grp, name_grp, y_ran)
                for mod, z in zip(model_states, z_states)], 0)
            for z_states in z_states_batches])
        for i, name in enumerate(name_grp):
            save_pickle(y_grp[..., i], f'{prefix}iSCALE_output/super_res_gene_expression/cnts-super/{name}.pickle')


def impute(
        embs, cnts, locs, radius, epochs, batch_size, prefix,
        n_states=1, load_saved=True, device='cuda', n_jobs=1,
        nbr_idx=None):

    single_cell = nbr_idx is not None

    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    if single_cell:
        # HistoOnco: targets are UCell pathway scores, already in [0, 1] and
        # rank-based; a single Atera capture has no inter-capture batch effect.
        # -> skip the cross-capture min-max normalization. L_pred is then plain
        # MSE in real UCell-score units, and prediction needs no un-scaling
        # (y_range = identity, mask_size = 1 below).
        cnts_min = np.zeros(cnts.shape[-1], dtype=np.float32)
        cnts_max = np.ones(cnts.shape[-1], dtype=np.float32)
    else:
        __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts) #norm here

    # mask = np.isfinite(embs).all(-1)
    # embs[~mask] = 0.0

    kwargs_list = [
            dict(
                x=embs, y=cnts, locs=locs, radius=radius,
                batch_size=batch_size, epochs=epochs, lr=1e-4,
                prefix=f'{prefix}states/{i:02d}/',
                load_saved=load_saved, device=device,
                nbr_idx=nbr_idx)
            for i in range(n_states)]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states
    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    model_list = [out[0] for out in out_list]
    dataset_list = [out[1] for out in out_list]
    # SpotDataset pools features over a disk of `mask_size` pixels, so per-pixel
    # predictions are scaled by 1/mask_size at inference. CellDataset does no
    # pooling (mask is None) -> mask_size = 1, i.e. no rescaling.
    mask = dataset_list[0].mask
    mask_size = 1 if mask is None else mask.sum()

    # embs[~mask] = np.nan
    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range = cnts_range / mask_size

    batch_size_row = 50
    n_batches_row = embs.shape[0] // batch_size_row + 1
    embs_batches = np.array_split(embs, n_batches_row)
    del embs
    predict(
            model_states=model_list, x_batches=embs_batches,
            name_list=names, y_range=cnts_range,
            prefix=prefix, device=device)
    # show_results(cnts_pred, names, prefix)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--epochs', type=int, default=None)  # e.g. 400
    parser.add_argument('--n-states', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--dist', type=int, default=100)
    parser.add_argument('--load-saved', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if SINGLE_CELL_MODE:
        # HistoOnco: single-cell-resolution pathway-score labels.
        # radius / --dist are unused here (no disk pooling, no daughter-capture
        # smoothing) but are kept in get_args so the documented command line
        # `python impute_integrated.py PREFIX --epochs=1000 ... --dist=...`
        # still parses unchanged.
        radius = None
        embs, cnts, locs, nbr_idx = get_data_cells(args.prefix)
    else:
        factor = 16
        radius = int(read_string(f'{args.prefix}radius.txt'))
        radius = radius / factor
        embs, cnts, locs = get_data_smooth(args.prefix, radius, args.dist)
        nbr_idx = None

    n_train = cnts.shape[0]
    batch_size = max(1, min(128, n_train // 16))

    impute(
            embs=embs, cnts=cnts, locs=locs, radius=radius,
            epochs=args.epochs, batch_size=batch_size,
            n_states=args.n_states, prefix=args.prefix,
            load_saved=args.load_saved,
            device=args.device, n_jobs=args.n_jobs,
            nbr_idx=nbr_idx)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
