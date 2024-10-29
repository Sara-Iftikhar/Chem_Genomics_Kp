
import os
import gc
import cv2
import warnings
import numpy as np
import pandas as pd
from Bio import SeqIO
import streamlit as st
from typing import Union, Any
import matplotlib.pyplot as plt
from collections import defaultdict

from easy_mpl import regplot, hist, plot
from SeqMetrics import RegressionMetrics

SAVE = True

def extract_colony(img, row, col, cell_width, cell_height):
    # Calculate the top-left corner of the grid cell
    start_x = col * cell_width
    start_y = row * cell_height

    # Crop the image using slicing
    crop_img = img[start_y:start_y + cell_height, start_x:start_x + cell_width]

    return crop_img

# Function to extract a colony based on row, column, and predefined dimensions
def extract_colony_using_dim(img, row, col, colony_dimensions):
    if (row, col) in colony_dimensions:
        start_x, start_y, width, height = colony_dimensions[(row, col)]
        colony_img = img[start_y:start_y + height, start_x:start_x + width]
        return colony_img
    else:
        st.error(f"No colony found at row {row}, column {col}")
        return None


def crop_img(path):

    # Step 1: Read the image
    image = cv2.imread(path)

    # Get the dimensions of the image (height, width, channels)
    h, w, _ = image.shape

    # Step 2: Define cropping boundaries
    # For example, to remove 10% from each side of the image
    top = int(0.015 * h)  # Remove 10% from the top
    bottom = int(0.989 * h)  # Remove 10% from the bottom
    left = int(0.01 * w)  # Remove 10% from the left
    right = int(0.99 * w)  # Remove 10% from the right

    # Step 3: Crop the image using NumPy slicing
    cropped_image = image[top:bottom, left:right]

    return cropped_image


def prepare_data(return_data=False, dependent_file_type='', input_file_path=f'data/unitigs_kp.rtab'):

    # Read the input rtab file
    df = pd.read_csv(input_file_path, delimiter='\t', index_col=0)
    reference_df = pd.read_csv(f'data/gene_presence_absence.Rtab', delimiter='\t', index_col=0)

    X = df.T

    reference_df = reference_df.T

    X = X.reindex(reference_df.index)

    del df
    gc.collect()

    # Read the target rtab file
    y = pd.read_csv(f'data/dependent_median_{dependent_file_type}_ML.csv')

    y.index = reference_df.index

    del reference_df
    gc.collect()


    if return_data:
        input_features = X.columns
        output_features = y.columns
        data = pd.concat([X, y], axis=1)
        return data, input_features, output_features
    return X, y


def generate_kmers(sequence, k):
    """
    Generate k-mers from a given sequence.
    """
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return kmers

def count_kmers_in_file(file_path, k, sample):
    """
    Count k-mers in a given file.
    """
    sample_name = os.path.splitext(sample)[0]
    kmer_counts = defaultdict(int)
    for record in SeqIO.parse(file_path, 'fasta'):
        kmers = generate_kmers(str(record.seq), k)
        for kmer in kmers:
            kmer_counts[kmer] += 1
    return {kmer: f"{sample_name}:{count}" for kmer, count in kmer_counts.items()}

def aggregate_kmer_counts(kmer_counts_list):
    """
    Aggregate k-mer counts from multiple samples.
    """
    aggregated_counts = defaultdict(list)
    for kmer_counts in kmer_counts_list:
        for kmer, count in kmer_counts.items():
            aggregated_counts[kmer].append(count)
    return aggregated_counts


# %%

def set_xticklabels(
        ax:plt.Axes,
        max_ticks:Union[int, Any] = 5,
        dtype = int,
        weight = "bold",
        fontsize:Union[int, float]=12,
        max_xtick_val=None,
        min_xtick_val=None,
        **kwargs
):
    return set_ticklabels(ax, "x", max_ticks, dtype, weight, fontsize,
                          max_tick_val=max_xtick_val,
                          min_tick_val=min_xtick_val,
                          **kwargs)


def set_yticklabels(
        ax:plt.Axes,
        max_ticks:Union[int, Any] = 5,
        dtype=int,
        weight="bold",
        fontsize:int=12,
        max_ytick_val = None,
        min_ytick_val = None,
        **kwargs
):
    return set_ticklabels(
        ax, "y", max_ticks, dtype, weight,
        fontsize=fontsize,
        max_tick_val=max_ytick_val,
        min_tick_val=min_ytick_val,
        **kwargs
    )


def set_ticklabels(
        ax:plt.Axes,
        which:str = "x",
        max_ticks:int = 5,
        dtype=int,
        weight="bold",
        fontsize:int=12,
        max_tick_val = None,
        min_tick_val = None,
        **kwargs
):
    ticks_ = getattr(ax, f"get_{which}ticks")()
    ticks = np.array(ticks_)
    if len(ticks)<1:
        warnings.warn(f"can not get {which}ticks {ticks_}")
        return

    if max_ticks:
        ticks = np.linspace(min_tick_val or min(ticks), max_tick_val or max(ticks), max_ticks)

    ticks = ticks.astype(dtype)

    getattr(ax, f"set_{which}ticks")(ticks)

    getattr(ax, f"set_{which}ticklabels")(ticks,
                                          weight=weight,
                                          fontsize=fontsize,
                                          **kwargs
                                          )
    return ax


def regression_plot(
        true,
        prediction,
        title,
        show:bool = True,
        hist_bins = 20,
        label="Source Prediction",
):

    hist_kws =  {"linewidth":0.5, "edgecolor":"k",
                 'bins': hist_bins}

    scatter_kws = {'marker': "o", 'edgecolors': 'black',
                   'linewidth':0.8, 'alpha': 0.6}

    ax = regplot(true,
                 prediction,
                 fill_color="orange",
                 line_color='dimgray',
                 scatter_kws=scatter_kws,
                 marginals=True, show=False,
                 hist_kws=hist_kws,
                 ax_kws=dict(xlabel=f"Experimental {label}",
                             ylabel=f'Predicted {label}'))

    metrics = RegressionMetrics(true, prediction)

    r2 = metrics.r2()
    rmse = metrics.rmse()
    mae = metrics.mae()

    ax.annotate(f'$R^2$= {round(r2, 3)}',
                xy=(0.95, 0.35),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12, weight="bold")
    ax.annotate(f'RMSE= {round(rmse, 3)}',
                xy=(0.95, 0.25),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12, weight="bold")

    ax.annotate(f'MAE= {round(mae, 3)}',
                xy=(0.95, 0.15),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12, weight="bold")

    if SAVE:
        label = label.replace('/', '_')
        plt.savefig(f"figures/{title}_{label}.png",
                    bbox_inches="tight",
                    dpi=600)
    return ax

# %%

def residual_plot(
        train_true,
        train_prediction,
        test_true,
        test_prediction,
        label="Prediction",
        train_color="orange",
        test_color="royalblue",
        show:bool = False
):
    fig, axis = plt.subplots(1, 2, sharey="all"
                             , gridspec_kw={'width_ratios': [2, 1]})
    test_y = test_true.reshape(-1, ) - test_prediction.reshape(-1, )
    train_y = train_true.reshape(-1, ) - train_prediction.reshape(-1, )
    train_hist_kws = dict(bins=20, linewidth=0.5,
                          edgecolor="k", grid=False, color=train_color,  # "#009E73"
                          orientation='horizontal')
    hist(train_y, show=False, ax=axis[1],
         label="Training", **train_hist_kws)
    plot(train_prediction, train_y, 'o', show=False,
         ax=axis[0],
         color=train_color,
         markerfacecolor=train_color,
         markeredgecolor="black", markeredgewidth=0.5,
         alpha=0.7, label="Training"
         )

    #****#
    test_hist_kws = dict(bins=40, linewidth=0.5,
                     edgecolor="k", grid=False,
                     color=test_color,
                     orientation='horizontal')
    hist(test_y, show=False, ax=axis[1],
         **test_hist_kws)

    set_xticklabels(axis[1], 3)

    plot(test_prediction, test_y, 'o', show=False,
         ax=axis[0],
         color=test_color,
         markerfacecolor=test_color,
         markeredgecolor="black", markeredgewidth=0.5,
         ax_kws=dict(
             xlabel=label,
             ylabel="Residual",
             legend_kws=dict(loc="upper left"),
         ),
         alpha=0.7, label="Test",
         )
    set_xticklabels(axis[0], 5)
    set_yticklabels(axis[0], 5)
    axis[0].axhline(0.0, color="black")
    plt.subplots_adjust(wspace=0.15)

    if show:
       plt.show()
    return


# def write_aggregated_kmers_to_files(aggregated_counts, output_prefix, num_files):
#     """
#     Write the aggregated k-mer counts to multiple files.
#     """
#     items = list(aggregated_counts.items())
#     chunk_size = len(items) // num_files
#     for i in range(num_files):
#         chunk = items[i * chunk_size: (i + 1) * chunk_size]
#         output_file = f"{output_prefix}_chunk_{i + 1}.txt"
#         with open(output_file, 'w') as out_file:
#             for kmer, counts in chunk:
#                 out_file.write(f"{kmer} | {' '.join(counts)}\n")

#     # If there are any remaining items, write them to a new file
#     remaining_items = items[num_files * chunk_size:]
#     if remaining_items:
#         output_file = f"{output_prefix}_chunk_{num_files + 1}.txt"
#         with open(output_file, 'w') as out_file:
#             for kmer, counts in remaining_items:
#                 out_file.write(f"{kmer} | {' '.join(counts)}\n")

def write_aggregated_kmers_to_files(aggregated_counts, output_file):
    """
    Write the aggregated k-mer counts to a single file.
    """
    with open(output_file, 'w') as out_file:
        for kmer, counts in aggregated_counts.items():
            out_file.write(f"{kmer} | {' '.join(counts)}\n")