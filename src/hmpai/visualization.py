from collections import defaultdict, Counter
from hmpai.behaviour.sat2 import match_on_event_name
from mne.viz import plot_topomap
from mne import Info
import numpy as np
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess
import captum
from hmpai.pytorch.utilities import DEVICE, save_tensor
from torch.utils.data import DataLoader
import torch
from hmpai.utilities import (
    MASKING_VALUE,
    calc_ratio,
    get_masking_index,
    get_masking_indices,
)
from tqdm.notebook import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind
from typing import List, Optional
import json
from rpy2.robjects import conversion, default_converter
from sklearn.preprocessing import StandardScaler
from pymer4.models import Lmer
import warnings
from hmpai.utilities import set_seaborn_style


def plot_max_activation_per_label(
    dataset: xr.Dataset,
    positions: Info,
    labels: list[str] = SAT1_STAGES_ACCURACY,
    save: bool = False,
) -> None:
    """
    Plots the maximum activation per label for a given dataset.

    Args:
        dataset (xr.Dataset): The dataset to plot.
        positions (Info): The positions of the sensors.
        labels (list[str], optional): The labels to plot. Defaults to SAT1_STAGES_ACCURACY.
        save (bool, optional): Whether to save the plot. Defaults to False.

    Returns:
        None
    """
    set_seaborn_style()
    f, ax = plt.subplots(nrows=3, ncols=len(labels), figsize=(8, 4))

    for i, label in enumerate(labels):
        # Get subset for label
        subset = dataset.sel(labels=label)
        n = len(subset.index)
        # Get maximum activation averaged over channels
        # TODO: Mean channels or sum channels? Maybe something else, highest n values?
        max_activation = abs(subset.analysis).sum("channels").argmax("samples")
        max_samples = subset.sel(samples=max_activation)
        mean_max_samples = max_samples.mean("index")

        ax[0, i].set_title(f"{label}\n(n={n})", fontsize=10)
        # Row titles
        if i == 0:
            ax[0, i].text(-0.30, 0, "EEG\nActivity", va="center", ha="left")
            ax[1, i].text(-0.30, 0, "Model\nAttention", va="center", ha="left")
            ax[2, i].text(-0.30, 0, "Combined", va="center", ha="left")

        # Raw EEG Activation
        plot_topomap(
            mean_max_samples.data,
            positions,
            axes=ax[0, i],
            show=False,
            cmap="Spectral_r",
            vlim=(np.min(mean_max_samples.data), np.max(mean_max_samples.data)),
            sensors=False,
            contours=6,
        )

        # Model activity
        abs_mean_max_analysis = mean_max_samples.analysis
        plot_topomap(
            abs_mean_max_analysis,
            positions,
            axes=ax[1, i],
            show=False,
            cmap="bwr",
            vlim=(np.min(abs_mean_max_analysis), np.max(abs_mean_max_analysis)),
            sensors=False,
            contours=6,
        )

        # Combined
        combined = mean_max_samples.data * abs_mean_max_analysis
        plot_topomap(
            combined,
            positions,
            axes=ax[2, i],
            show=False,
            cmap="Spectral_r",
            vlim=(np.min(combined), np.max(combined)),
            sensors=False,
            contours=6,
        )
    plt.tight_layout()
    if save:
        plt.savefig("img/max_activation_per_label.png", transparent=True)
    else:
        plt.show()


def plot_mean_activation_per_label(dataset: xr.Dataset, positions: Info) -> None:
    """
    Plots the mean activation per label.

    Parameters:
    -----------
    dataset : xr.Dataset
        The dataset containing the activations.
    positions : dict
        The positions of the sensors.

    Returns:
    --------
    None
    """
    f, ax = plt.subplots(nrows=1, ncols=len(SAT1_STAGES_ACCURACY), figsize=(6, 3))

    for i, label in enumerate(SAT1_STAGES_ACCURACY):
        subset = dataset.sel(labels=label)
        n = len(subset.index)

        mean_activation = subset.mean(["index", "samples"]).data
        ax[i].set_title(f"{label}\n(n={n})", fontsize=10)
        plot_topomap(
            mean_activation,
            positions,
            axes=ax[i],
            show=False,
            cmap="Spectral_r",
            vlim=(np.min(mean_activation), np.max(mean_activation)),
            sensors=False,
            contours=6,
        )

    plt.tight_layout()
    plt.show()


def plot_single_trial_activation(sample: xr.Dataset, positions: Info) -> None:
    """
    Plots the raw EEG activation, model attention, and combined activation for a single trial.

    Args:
        sample (xr.Dataset): The EEG data for a single trial.
        positions (Info): The positions of the EEG sensors.

    Returns:
        None
    """
    nan_index = np.isnan(sample.data.where(sample.data != 999)).argmax(
        dim=["samples", "channels"]
    )
    nan_index = nan_index["samples"].item()
    if nan_index == 0:
        return

    f, ax = plt.subplots(nrows=3, ncols=nan_index, figsize=(nan_index, 3))
    combined = sample.data[0:nan_index, :] * sample.analysis[0:nan_index, :]
    for i in range(nan_index):
        # Raw EEG Activation
        ax[0, i].set_title(f"Sample {i}", fontsize=10)
        # Row titles
        if i == 0:
            ax[0, i].text(-0.35, 0, "EEG\nActivity", va="center", ha="left")
            ax[1, i].text(-0.35, 0, "Model\nAttention", va="center", ha="left")
            ax[2, i].text(-0.35, 0, "Combined", va="center", ha="left")
        plot_topomap(
            sample.data[i, :],
            positions,
            axes=ax[0, i],
            show=False,
            cmap="Spectral_r",
            vlim=(
                np.min(sample.data[0:nan_index, :]),
                np.max(sample.data[0:nan_index, :]),
            ),
            sensors=False,
            contours=6,
        )

        # Model activity
        plot_topomap(
            sample.analysis[i, :],
            positions,
            axes=ax[1, i],
            show=False,
            cmap="bwr",
            vlim=(
                np.min(sample.analysis[0:nan_index, :]),
                np.max(sample.analysis[0:nan_index, :]),
            ),
            sensors=False,
            contours=6,
        )

        # Combined
        plot_topomap(
            combined[i, :],
            positions,
            axes=ax[2, i],
            show=False,
            cmap="Spectral_r",
            vlim=(np.min(combined), np.max(combined)),
            sensors=False,
            contours=6,
        )
    plt.tight_layout()
    plt.show()


def plot_model_attention_over_stage_duration(
    dataset: xr.Dataset, labels: list[str] = SAT1_STAGES_ACCURACY, save: bool = False
) -> None:
    """
    Plots the model attention over stage duration for the given dataset.

    Parameters:
    dataset (xr.Dataset): The dataset containing the data to be plotted.
    labels (list[str], optional): The labels to plot. Defaults to SAT1_STAGES_ACCURACY.
    save (bool, optional): Whether to save the plot. Defaults to False.

    Returns:
    None
    """
    set_seaborn_style()
    f, ax = plt.subplots(
        nrows=1,
        ncols=len(labels),
        figsize=(8, 3),
        sharey=True,
        sharex=True,
    )

    time_points = np.linspace(0, 100, 100)
    for i, label in enumerate(labels):
        subset = dataset.sel(labels=label)
        nan_indices = np.isnan(subset.data.where(subset.data != MASKING_VALUE)).argmax(
            dim=["samples", "channels"]
        )
        interpolated = []
        for sample, nan_index in zip(subset.analysis, nan_indices["samples"]):
            if nan_index.item() < 3:
                continue
            sequence = sample.mean(dim="channels")[0 : nan_index.item()]
            if len(sequence) == 0:
                continue
            origin_time_points = np.linspace(0, 100, num=len(sequence))
            interpolated_sequence = np.interp(time_points, origin_time_points, sequence)
            interpolated.append(abs(interpolated_sequence))
        ax[i].plot(np.mean(interpolated, axis=0))
        ax[i].set_facecolor("white")
        ax[i].set_title(f"{label}", fontsize=10)
        ax[i].set_yticks(np.arange(0.0, 0.06, 0.01))
    # ax[2].text(0, -0.005, 'Linear interpolation\nof stage length', va='bottom', ha='center')
    ax[0].set_ylabel("Model Attention")
    ax[2].set_xlabel("Stage duration (%)")
    plt.tight_layout()
    if save:
        plt.savefig("img/model_attention_over_stage_duration.png", transparent=True)
    else:
        plt.show()


def plot_confusion_matrix(
    true: torch.Tensor, pred: torch.Tensor, labels: list, save: bool = False
) -> None:
    sns.set_style("ticks")
    sns.set_context("paper")
    print(sum(true == pred).item() / len(pred))
    cm = confusion_matrix(true, pred, normalize="true")
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="OrRd")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(0.5, len(labels) + 0.5), labels=labels, rotation=90)
    plt.yticks(ticks=np.arange(0.5, len(labels) + 0.5), labels=labels, rotation=0)
    if save:
        plt.savefig("img/confusion_matrix.png", transparent=True, bbox_inches="tight")
    else:
        plt.show()


def p_to_asterisk(pvalue: float):
    """
    Converts a p-value to an asterisk representation based on significance levels.

    Args:
        pvalue (float): The p-value to convert.

    Returns:
        str: The asterisk representation based on the significance level of the p-value.
            - "****" for p-value <= 0.0001
            - "***" for p-value <= 0.001
            - "**" for p-value <= 0.01
            - "*" for p-value <= 0.05
            - "ns" for p-value > 0.05
    """
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


def generate_table(
    accs: List[List[float]], f1s: List[List[float]], categories: List[str]
):
    table_start = """
\\begin{table}[H]
  \\centering
  \\begin{tabular}{@{}lll@{}} \\toprule
      & Accuracy             & F1-Score             \\\\ \\midrule"""
    table_end = """
  \\end{tabular}
  \\caption{CAPTION HERE}
\\end{table}"""

    table = ""
    for acc, f1, cat in zip(accs, f1s, categories):
        table += f"\n  {cat} & {np.mean(acc):.2f}\% (SD {np.std(acc):.2f}) & {np.mean(f1):.2f}\% (SD {np.std(f1):.2f}) \\\\"
        if cat == categories[-1]:
            table += " \\bottomrule"

    return table_start + table + table_end


def plot_performance(
    accs: List[List[float]],
    f1s: List[List[float]],
    categories: List[str],
    cat_name: str,
    legend_pos: str = "lower right",
    ylim=(0, 100),
):
    n_obs = len(accs[0])
    df = pd.DataFrame(
        {
            "Accuracy": [acc for sublist in accs for acc in sublist],
            "F1-score": [f1 for sublist in f1s for f1 in sublist],
            "category": [category for category in categories for _ in range(n_obs)],
        }
    )
    df = df.melt(
        id_vars="category",
        value_vars=("Accuracy", "F1-score"),
        var_name="metric",
        value_name="value",
    )
    set_seaborn_style()

    # fig, axes = plt.subplots(1, 2, figsize=(len(accs) * 2, len(accs)), gridspec_kw={"width_ratios": [2, 1]})
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [2, 1]}
    )
    sns.violinplot(
        data=df, x="category", y="value", hue="metric", split=True, ax=axes[0]
    )
    axes[0].set_ylabel("Metric value (%)")
    axes[0].set_xlabel(cat_name)
    axes[0].set_ylim(ylim)
    if legend_pos == "lower right":
        sns.move_legend(
            axes[0], "lower right", bbox_to_anchor=(1.0, 0.0), title="Metric"
        )
    else:
        sns.move_legend(
            axes[0], "upper right", bbox_to_anchor=(1.0, 1.0), title="Metric"
        )
    sns.despine()
    means = df.groupby(["category", "metric"], sort=False).mean().reset_index()
    print(means)
    acc_means = means[means.metric == "Accuracy"].value.to_numpy()
    f1_means = means[means.metric == "F1-score"].value.to_numpy()
    # TODO: To abs or not to abs
    acc_diffs = acc_means[:, np.newaxis] - acc_means
    f1_diffs = -(f1_means[:, np.newaxis] - f1_means)

    mask = np.tril(np.ones_like(acc_diffs), k=-1)
    mask_upper = np.tril(np.ones_like(acc_diffs), k=0)
    acc_diffs[mask == 0] = np.nan
    f1_diffs[mask_upper != 0] = np.nan

    # Remove values that are not interesting to compare
    if cat_name == "Model-Sampling Frequency":
        f1_diffs[0, 3] = np.nan
        f1_diffs[1, 2] = np.nan
        acc_diffs[3, 0] = np.nan
        acc_diffs[2, 1] = np.nan
    blue_palette = sns.light_palette(color=sns.color_palette()[0], as_cmap=True)
    orange_palette = sns.light_palette(color=sns.color_palette()[1], as_cmap=True)
    # If only one value occurs, it is seen as 'low', reverse this to make colors show up
    if len(categories) == 2:
        blue_palette = blue_palette.reversed()
        orange_palette = orange_palette.reversed()
    sns.heatmap(
        acc_diffs,
        annot=True,
        fmt=".3f",
        xticklabels=categories,
        yticklabels=categories,
        cmap=blue_palette,
        cbar=False,
        ax=axes[1],
    )

    sns.heatmap(
        f1_diffs,
        annot=True,
        fmt=".3f",
        xticklabels=categories,
        yticklabels=categories,
        cmap=orange_palette,
        cbar=False,
        ax=axes[1],
    )
    ax2 = axes[1].twinx()
    ax2.set_ylim(axes[1].get_ylim())
    ax2.set_yticks(axes[1].get_yticks())
    ax2.set_yticklabels(axes[1].get_yticklabels(), rotation=90, va="center")
    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    axes[1].set_yticks([])

    # axes[1].set_ylabel("From")
    # axes[0].set_title("Distribution of folds")
    # axes[1].set_title("Differences in performance")
    axes[1].set_xlabel("Performance change (%)")
    plt.tight_layout()
    plt.savefig(f"img/{cat_name}.png")
    plt.show()


def plot_performance_ttest(
    accs: List[List[float]],
    f1s: List[List[float]],
    categories: List[str],
    cat_name: str,
    ttest_origins: Optional[List[Optional[int]]] = None,
) -> None:
    """
    Plots the performance metrics (accuracy and F1-score) for different categories.

    Args:
        accs (List[List[float]]): List of lists containing accuracy values for each category.
        f1s (List[List[float]]): List of lists containing F1-score values for each category.
        categories (List[str]): List of category names.
        cat_name (str): Name of the category variable.
        ttest_origins (Optional[List[Optional[int]]], optional): List of indices of categories that are compared.
            The length should be len(categories) - 1. If a value in ttest_origins equals None, then it is skipped.
            Defaults to None.

    Returns:
        None
    """
    n_obs = len(accs[0])

    df = pd.DataFrame(
        {
            "Accuracy": [acc for sublist in accs for acc in sublist],
            "F1-score": [f1 for sublist in f1s for f1 in sublist],
            "category": [category for category in categories for _ in range(n_obs)],
        }
    )
    df = df.melt(
        id_vars="category",
        value_vars=("Accuracy", "F1-score"),
        var_name="metric",
        value_name="value",
    )
    set_seaborn_style()
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    grid = sns.catplot(
        data=df, x="category", y="value", hue="metric", kind="violin", split=True
    )
    grid.set(ylim=(0.8, 1.05))
    # grid.axes[0][0].set_title("Accuracy")
    grid.axes[0][0].set_ylabel("Metric value")
    # grid.axes[0][1].set_title("f1-score")
    grid.set(xlabel=cat_name, yticks=np.arange(0.8, 1.05, 0.05))
    # grid._legend.remove()
    sns.move_legend(grid, "lower right", bbox_to_anchor=(1.0, 0.1), title="Metric")
    # sns.barplot(df, x="category", y="acc", ax=axes[0], hue="category")
    # axes[0].set_ylim((0.84, 1.0))
    # axes[0].set_ylabel("Accuracy")
    # axes[0].set_xlabel(cat_name)

    # sns.catplot(data=df, x="category", y="f1", ax=axes[1], hue="category")
    # # sns.barplot(df, x="category", y="f1", ax=axes[1], hue="category")
    # axes[1].set_ylim((0.84, 1.0))
    # axes[1].set_ylabel("f1-score")
    # axes[1].set_xlabel(cat_name)

    # for i, ax in enumerate(grid.axes[0]):
    #     for j, cat in enumerate(categories):
    #         if j == 0:
    #             continue
    #         x1, x2 = 0, j
    #         y, h, col = ax.get_ylim()[1] - (0.06 - 0.015 * j), 0.005, "k"
    #         ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    #         if i == 0:
    #             significance = ttest_rel(accs[j], accs[0])
    #         else:
    #             significance = ttest_rel(f1s[j], f1s[0])
    #         print(f"{ax.get_title()}, category {cat} vs {categories[0]}")
    #         print(significance)
    #         ax.text(
    #             (x1 + x2) * 0.5,
    #             y + h,
    #             p_to_asterisk(significance[1]),
    #             ha="center",
    #             va="bottom",
    #             color=col,
    #         )
    for j, cat in enumerate(categories):
        if j == 0:
            continue
        if ttest_origins is not None:
            if ttest_origins[j - 1] is None:
                continue
            x1, x2 = ttest_origins[j - 1], j
        else:
            x1, x2 = 0, j
        y, h, col = grid.axes[0][0].get_ylim()[1] - (0.06 - 0.015 * j), 0.005, "k"
        grid.axes[0][0].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        acc_significance = ttest_rel(accs[j], accs[x1])
        f1_significance = ttest_rel(f1s[j], f1s[x1])
        print(f"Accuracy, category {cat} vs {categories[x1]}")
        print(acc_significance)
        print(f"F1-score, category {cat} vs {categories[x1]}")
        print(f1_significance)

        grid.axes[0][0].text(
            (x1 + x2) * 0.5,
            y + h,
            f"{p_to_asterisk(acc_significance[1])}  {p_to_asterisk(f1_significance[1])}",
            ha="center",
            va="bottom",
            color=col,
        )
    plt.tight_layout()
    plt.show()
    # sns.despine()


def plot_performance_from_file(
    path,
    conditions,
    cat_name,
    do_generate_table=False,
    legend_pos="lower right",
    ylim=(0.0, 100.0),
):
    res = defaultdict(lambda: defaultdict(list))
    if type(path) is list:
        for i, p in enumerate(path):
            with open(p) as f:
                data = json.load(f)
            for k, test_results in data.items():
                for fold in test_results:
                    res[conditions[int(i)]]["accuracy"].append(fold["accuracy"] * 100)
                    res[conditions[int(i)]]["f1"].append(
                        fold["weighted avg"]["f1-score"] * 100
                    )
    else:
        with open(path) as f:
            data = json.load(f)
        for k, test_results in data.items():
            for fold in test_results:
                res[conditions[int(k)]]["accuracy"].append(fold["accuracy"] * 100)
                res[conditions[int(k)]]["f1"].append(
                    fold["weighted avg"]["f1-score"] * 100
                )

    categories = []
    accs = []
    f1s = []
    for k, v in res.items():
        categories.append(k)
        accs.append(v["accuracy"])
        f1s.append(v["f1"])
    if do_generate_table:
        print(generate_table(accs, f1s, categories))
    plot_performance(accs, f1s, categories, cat_name, legend_pos=legend_pos, ylim=ylim)
    return res


def plot_predictions_on_epoch(
    epoch: torch.Tensor,
    true: torch.Tensor,
    labels: list[str],
    window_size: int,
    model: torch.nn.Module,
    smoothing: bool = False,
    sequence: bool = False,
    random_perm: bool = False,
    save_tensors: bool = False,
    save: bool = False,
):
    epoch = epoch.clone()

    def smooth_predictions(predictions, window_size):
        smoothed = np.copy(predictions)
        for i in range(window_size // 2, len(predictions) - window_size // 2):
            smoothed[i] = np.mean(
                predictions[i - window_size // 2 : i + window_size // 2 + 1], axis=0
            )
        return smoothed

    set_seaborn_style()
    empty = np.full((epoch.size()[0], epoch.size()[0], len(labels)), np.nan)
    # empty = np.zeros((epoch.size()[0], len(labels)))
    rt_idx = get_masking_index(epoch)
    # Normalize probability labels
    # true = true[:rt_idx, 1:]
    # true = true / true.sum()
    plot_true = true is not None
    if plot_true:
        true = true[:rt_idx, 1:]

    if not sequence:
        slices = get_padded_slices(epoch, window_size, include_start_end=False)
        stacked = torch.stack(slices).to(DEVICE)
        pred = model(stacked)
        # Dim = 2 works for proba prediction, probably should be dim=1 for class prediction?
        pred = torch.nn.Softmax(dim=2)(pred)
    else:
        pred = model(epoch.unsqueeze(0).to(DEVICE))
        pred = torch.nn.Softmax(dim=2)(pred)
        pred = pred / pred.sum(dim=2, keepdim=True)
    pred = pred.cpu().detach().numpy()
    if smoothing:
        pred = smooth_predictions(pred, window_size)

    if not sequence:
        for i, prediction in enumerate(pred):
            # TODO: Handle global pool case
            # empty[i,:] = prediction
            empty[i, i : i + window_size, :] = prediction
            # empty[i + window_size // 2, :] = prediction
        empty = np.nanmean(empty, axis=0)
    else:
        empty = pred.squeeze()

    # print(true)
    nrows = 3 if random_perm else 2
    fig, ax = plt.subplots(nrows, 1, figsize=(6, 9), dpi=100)
    # fig, ax = plt.subplots(nrows, 1, sharex=True)
    ax[0].set_ylabel("HMP")
    ax[0].set_xlabel("Samples")
    ax[1].set_ylabel("S4")
    # plt.setp(ax, ylim=(0, 0.1))
    fig.supylabel("Probability")
    fig.supxlabel("Samples")
    # HMP Probability
    if plot_true:
        for i in range(0, true.shape[1]):
            sns.lineplot(
                x=range(len(true[:rt_idx, i])),
                y=true[:rt_idx, i],
                ax=ax[0],
                color=sns.color_palette()[i + 1],
                label=labels[i + 1],
                legend=False,
            )
    if save_tensors:
        if plot_true:
            save_tensor(true[:rt_idx], "hmp_pred.csv")
        save_tensor(pred.squeeze()[:rt_idx, 1:], "mamba_pred.csv")
    # Model probability
    for i in range(1, empty.shape[1]):
        sns.lineplot(
            x=range(len(empty[:rt_idx, i])),
            y=empty[:rt_idx, i],
            ax=ax[1],
            color=sns.color_palette()[i],
            label=labels[i],
            legend=False,
        )
    if random_perm:
        n_shuffles = 100
        shuffled_preds = np.full((n_shuffles, epoch.size()[0], len(labels)), np.nan)

        for i_shuf in range(n_shuffles):
            empty = np.full((epoch.size()[0], epoch.size()[0], len(labels)), np.nan)
            perm = torch.randperm(rt_idx)
            epoch[:rt_idx] = epoch[perm]
            if not sequence:
                slices = get_padded_slices(epoch, window_size, include_start_end=False)
                stacked = torch.stack(slices).to(DEVICE)
                pred = model(stacked)
                # Dim = 2 works for proba prediction, probably should be dim=1 for class prediction?
                pred = torch.nn.Softmax(dim=2)(pred)
            else:
                pred = model(epoch.unsqueeze(0).to(DEVICE))
                pred = torch.nn.Softmax(dim=2)(pred)
            pred = pred.cpu().detach().numpy()
            if smoothing:
                pred = smooth_predictions(pred, window_size)

            if not sequence:
                for i, prediction in enumerate(pred):
                    # TODO: Handle global pool case
                    # empty[i,:] = prediction
                    empty[i, i : i + window_size, :] = prediction
                    # empty[i + window_size // 2, :] = prediction
                empty = np.nanmean(empty, axis=0)
            else:
                empty = pred.squeeze()
            shuffled_preds[i_shuf, : empty.shape[0], :] = empty
        shuffled_preds = shuffled_preds.mean(axis=0)
        shuffled_preds = shuffled_preds / shuffled_preds.sum(axis=1, keepdims=True)
        if save_tensors:
            save_tensor(shuffled_preds[:rt_idx, 1:], "shuffled_pred.csv")

        # Shuffled probability
        ax[2].set_ylabel("Shuffled")
        # ax[1].set_ylim((0, 0.1))
        # ax[2].set_ylim((0, 0.1))
        for i in range(1, shuffled_preds.shape[1]):
            sns.lineplot(
                x=range(len(shuffled_preds[:rt_idx, i])),
                y=shuffled_preds[:rt_idx, i],
                ax=ax[2],
                color=sns.color_palette()[i],
                label=labels[i],
                legend=False,
            )
    # plt.ylabel("Probability")
    handles, _ = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels[1:])
    plt.tight_layout()
    if save:
        plt.savefig("../../img/drieluik.svg", transparent=True)
    plt.show()


def plot_stage_predictions(
    epoch: torch.Tensor, labels: list[str], window_size: int, model: torch.nn.Module
):
    set_seaborn_style()
    empty = np.zeros((epoch.size()[0], len(labels)))
    rt_idx = get_masking_index(epoch)

    slices = get_padded_slices(epoch, window_size, include_start_end=False)
    stacked = torch.stack(slices).to(DEVICE)
    # Pad to the size the model is used to
    stacked = torch.nn.functional.pad(
        stacked, (0, 0, 0, 100 - window_size), value=MASKING_VALUE
    )
    try:
        pred = model(stacked)
    except Exception:
        print("Oopsie")
        return
    pred = torch.nn.Softmax(dim=1)(pred)
    pred = pred.cpu().detach().numpy()
    # if smoothing:
    #     pred = smooth_predictions(pred, window_size)

    for i, prediction in enumerate(pred):
        empty[i, :] = prediction
        # empty[i + window_size // 2, :] = prediction

    fig, ax = plt.subplots()
    for i in range(0, empty.shape[1]):
        sns.lineplot(
            x=range(len(empty[:, i])),
            y=empty[:, i],
            ax=ax,
            color=sns.color_palette()[i],
            label=labels[i],
        )
    # sns.lineplot(empty[:, 1:], ax=ax)
    # label=SAT_CLASSES_ACCURACY[1:]
    ax.legend()
    plt.xlim(0, rt_idx + (window_size // 2))
    plt.ylim(0, 1.0)
    plt.xlabel("Samples")
    plt.ylabel("Softmax probability")
    plt.show()


def get_padded_slices(
    epoch: torch.Tensor, window_size: int, include_start_end: bool = True
):
    # Get index where epoch ends
    rt_idx = get_masking_index(epoch)
    slices = []

    # Right-pad beginning elements
    if include_start_end:
        for start in range(-window_size + 1, 0):
            padded_slice = torch.full(
                (window_size, epoch.size(1)), MASKING_VALUE, dtype=torch.float32
            )
            if start < 0:
                padded_slice[0 : window_size - -start, :] = epoch[
                    0 : window_size - -start, :
                ]
            slices.append(padded_slice)

    # Normal slices
    for start in range(rt_idx - window_size + 1):
        slices.append(epoch[start : start + window_size, :])

    # Right-pad end
    if include_start_end:
        for start in range(rt_idx - window_size + 1, rt_idx - 1):
            padded_slice = torch.full(
                (window_size, epoch.size(1)), MASKING_VALUE, dtype=torch.float32
            )
            if start + window_size > rt_idx:
                valid_length = rt_idx - start + 1
                padded_slice[:valid_length, :] = epoch[start : start + valid_length, :]
            slices.append(padded_slice)

    return slices


def predict_with_auc(
    model: torch.nn.Module,
    loader: DataLoader,
    info_to_keep: list[str],
    labels: list[str],
):
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)
    data_list = []
    for batch in loader:
        data = {info_key: batch[2][0][info_key] for info_key in info_to_keep}
        pred = model(batch[0].to(DEVICE))
        rt_indices = get_masking_indices(batch[0])
        data["rt_index_samples"] = rt_indices
        pred = torch.nn.Softmax(dim=2)(pred)
        # Disregard negative class
        # pred[:,:,1:] = pred[:,:,1:] / (1 - pred[:,:,0]).unsqueeze(-1)
        # pred[:,:,1:] = pred[:,:,1:] / pred[:,:,0].unsqueeze(-1)
        batch_aucs = torch.sum(pred, dim=1)
        batch_aucs = batch_aucs.cpu().detach()

        # Append AUCs
        for i, label in enumerate(labels):
            data[label + "_auc"] = batch_aucs[:, i]
        data_list.append(pd.DataFrame(data))
    data = pd.concat(data_list)
    return data


def plot_median_split_error_rate(data: pd.DataFrame, operation: str):
    set_seaborn_style()
    data["ratio"] = (data[f"{operation}_auc"] / 256) / data["rt_x"]
    median = data["ratio"].median()
    data["above_median"] = data["ratio"] >= median
    plt.figure(figsize=(4, 6))
    plot = sns.barplot(
        data=data, x="SAT", y="response", hue="above_median", legend=True
    )
    plt.ylabel("Proportion of correct responses")
    plt.xlabel(f"Below or above median of {operation} AUC/RT\n(Median: {median:.4f})")
    # plt.xticks(ticks=[0,1], labels=["Below", "Above"])
    plt.ylim(0, 1)
    plt.plot()


def plot_ratio_auc_over_RT(data: pd.DataFrame, operation: str):
    set_seaborn_style()
    # Divide by sampling frequency, change if sampling frequency changes
    data["ratio"] = (data[f"{operation}_auc"] / 100) / data["rt_x"]
    bins = np.linspace(0, 1, 11)
    labels = [f"{bin:.1f},{bin + 0.1:.1f}" for bin in bins]
    data["ratio_bin"] = pd.cut(data["ratio"], bins=bins)
    # data['ratio_bin'] = pd.qcut(data['ratio'], 5)

    data_sp = data[data["SAT"] == "speed"]
    data_acc = data[data["SAT"] == "accuracy"]

    crosstab_sp = pd.crosstab(
        data_sp["ratio_bin"], data_sp["response"], normalize="index"
    )
    crosstab_acc = pd.crosstab(
        data_acc["ratio_bin"], data_acc["response"], normalize="index"
    )

    crosstab_sp["SAT"] = "speed"
    crosstab_acc["SAT"] = "accuracy"
    crosstab = pd.concat([crosstab_sp, crosstab_acc])
    crosstab_long = crosstab.reset_index().melt(
        id_vars=["ratio_bin", "SAT"],
        value_vars=[0, 1],
        var_name="response",
        value_name="proportion",
    )

    plt.figure()
    plot = sns.barplot(
        x="ratio_bin",
        y="proportion",
        hue="SAT",
        data=crosstab_long[crosstab_long.response == 1],
        errorbar=("ci", 95),
    )
    plot.set_xticklabels(labels=labels, rotation=45)
    plt.xlabel(f"AUC of {operation} / RT, divided into 10 bins from 0 to 1")
    plt.ylabel("Proportion of correct responses")
    plt.ylim(0, 1)
    plt.xlim(-0.5, 9.5)
    plt.plot()


def plot_ratio_true_over_RT(data: pd.DataFrame, operation: str):
    set_seaborn_style()
    # Divide by sampling frequency, change if sampling frequency changes
    data["ratio"] = (data[f"{operation}_true_samples"] / 100) / data["rt_x"]
    bins = np.linspace(0, 1, 11)
    labels = [f"{bin:.1f},{bin + 0.1:.1f}" for bin in bins]
    data["ratio_bin"] = pd.cut(data["ratio"], bins=bins, include_lowest=True)
    # data['ratio_bin'] = pd.qcut(data['ratio'], 5)

    data_sp = data[data["SAT"] == "speed"]
    data_acc = data[data["SAT"] == "accuracy"]

    crosstab_sp = pd.crosstab(
        data_sp["ratio_bin"], data_sp["response"], normalize="index"
    )
    crosstab_acc = pd.crosstab(
        data_acc["ratio_bin"], data_acc["response"], normalize="index"
    )

    crosstab_sp["SAT"] = "speed"
    crosstab_acc["SAT"] = "accuracy"
    crosstab = pd.concat([crosstab_sp, crosstab_acc])
    crosstab_long = crosstab.reset_index().melt(
        id_vars=["ratio_bin", "SAT"],
        value_vars=[0, 1],
        var_name="response",
        value_name="proportion",
    )

    plt.figure()
    plot = sns.barplot(
        x="ratio_bin",
        y="proportion",
        hue="SAT",
        data=crosstab_long[crosstab_long.response == 1],
        errorbar=("ci", 95),
    )
    plot.set_xticklabels(labels=labels, rotation=45)
    plt.xlabel(f"HMP-predicted {operation} / RT, divided into 10 bins from 0 to 1")
    plt.ylabel("Proportion of correct responses")
    plt.ylim(0, 1)
    plt.xlim(-0.5, 9.5)
    plt.plot()


def show_lmer(labels: list[str], data: pd.DataFrame, formula: str):
    warnings.filterwarnings("ignore")
    data = data.copy()
    auc_columns = [
        column
        for column in data.columns
        if column.endswith("_auc") and not column.startswith("negative")
    ]
    # auc_columns = [column for column in data.columns if column.endswith("_auc")]
    # data[auc_columns] = (data[auc_columns] - data[auc_columns].min()) / (data[auc_columns].max() - data[auc_columns].min())
    data["sum_auc"] = data[auc_columns].sum(axis=1)
    # data['rt_x_norm'] = (data['rt_x'] - data['rt_x'].min()) / (data['rt_x'].max() - data['rt_x'].min())
    # Normalized operation certainty
    # for label in labels[1:]:
    #     data[f"{label}_ratio"] = data[f"{label}_auc"] / data["sum_auc"]
    for label in labels[1:]:
        data[f"{label}_ratio"] = data[f"{label}_auc"] / data["rt_x"]
    # ratio_columns = [column for column in data.columns if column.endswith("_ratio")]
    # data[ratio_columns] = (data[ratio_columns] - data[ratio_columns].min()) / (
    #     data[ratio_columns].max() - data[ratio_columns].min()
    # )

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    set_seaborn_style()
    plt.tight_layout()
    plt.setp(ax, ylabel="P(response == 1)")
    # plt.setp(ax, xlim=(0, 1), ylabel="P(response == 1)")

    plot_loc = {
        "encoding": ax[0][0],
        "decision": ax[0][1],
        "confirmation": ax[1][0],
        "response": ax[1][1],
    }
    for label in labels[1:]:
        print(f"---{label.upper()}---")
        # data["ratio"] = (data[f"{label}_auc"]) / data["rt_x_norm"]
        # data["ratio"] = (data[f"{label}_auc"]) / data["sum_auc"]

        data["participant"] = data["participant"].astype("category")

        if "?r" in formula:
            label_tag = f"{label}_ratio"
            label_formula = formula.replace("?r", label_tag)
        elif "?a" in formula:
            label_tag = f"{label}_auc"
            label_formula = formula.replace("?a", label_tag)
        else:
            label_tag = "ratio"
            label_formula = formula
        with conversion.localconverter(default_converter):
            lmer_model = Lmer(label_formula, data=data, family="binomial")
            result = lmer_model.fit()

            # Generate a range of ratio values
        print(lmer_model.summary())
        # TODO: This was to predict probability of ratio over a large range! Should probably use this instead of real data
        # ratio_values = np.linspace(
        #     data[label_tag].min(), data[label_tag].max(), len(data)
        # )
        # np.random.shuffle(ratio_values)

        # Create DataFrames for predictions for both conditions
        predict_df = data[["response", label_tag, "SAT", "participant"]]
        # predict_df[label_tag] = ratio_values
        with conversion.localconverter(default_converter):
            predict_df["predicted_prob"] = lmer_model.predict(
                predict_df, skip_data_checks=True, verify_predictions=False
            )

        plot_data = predict_df

        # Create the plot
        # plot_loc[label].set_xlabel(f'Ratio of {label} AUC over RT')
        plot_loc[label].set_xlabel(f"Normalized {label} AUC")
        # plot_loc[label].set_xlabel(f'Proportion of trial predicted as {label}')
        sns.lineplot(
            data=plot_data,
            x=label_tag,
            y="predicted_prob",
            hue="SAT",
            ax=plot_loc[label],
        )
        sns.scatterplot(
            data=data[data["SAT"] == "speed"],
            x=label_tag,
            y="response",
            alpha=0.1,
            label="Actual Data Speed",
            ax=plot_loc[label],
        )
        sns.scatterplot(
            data=data[data["SAT"] == "accuracy"],
            x=label_tag,
            y="response",
            alpha=0.1,
            label="Actual Data Accuracy",
            ax=plot_loc[label],
        )

        # Customize the plot
        plt.legend(title="Condition")
    plt.show()


def plot_eeg(epoch: torch.Tensor):
    set_seaborn_style()
    plt.figure(figsize=(6, 2))
    epoch = epoch[: get_masking_index(epoch)]
    for channel in range(epoch.shape[1]):
        sns.lineplot(epoch[:, channel])
    plt.xlabel("Samples")
    plt.ylabel("Normalized EEG")
    plt.tight_layout()
    plt.savefig("../../img/eeg.svg", transparent=True)
    plt.show()


def plot_epoch(item: tuple, title: str = ""):
    # Input is (data, labels, ... (info))
    data = item[0]
    labels = item[1]

    # Plot 1 channel and all (?) labels
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.title(title)

    ax[0].set_ylim((-5, 5))

    sns.lineplot(data[:, 0], ax=ax[0])
    for i in range(labels.shape[-1]):
        if i == 0:
            sns.lineplot(labels[:, i] * 0.1, ax=ax[1])
        else:
            sns.lineplot(labels[:, i], ax=ax[1])

    plt.plot()


def add_significance_annotations(p_value, group1, group2, y_position, ax):
    """Adds significance stars based on p-value."""
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"  # Not significant

    # Plot the annotation
    x1, x2 = group1, group2
    ax.plot(
        [x1, x1, x2, x2],
        [y_position, y_position * 1.025, y_position * 1.025, y_position],
        lw=1.0,
        color="black",
        alpha=1.0,
    )
    ax.text(
        (x1 + x2) / 2,
        y_position * 1.025,
        significance,
        ha="center",
        va="bottom",
        color="black",
        alpha=1.0,
    )


def plot_tertile_split(
    data: pd.DataFrame,
    column: str,
    conditions: list[str],
    calc_tertile_over_condition: bool = False,
    normalize: str = "auc",
):
    pd.options.mode.chained_assignment = None
    set_seaborn_style()
    data = data.copy()

    # Also do ratio here? Does not make sense since for non-confirmation operations we are certain that they exist?
    if column.endswith("_ratio"):
        if normalize == "auc":
            auc_columns = [
                column
                for column in data.columns
                if column.endswith("_auc") and not column.startswith("negative")
            ]
            data["sum_auc"] = data[auc_columns].sum(axis=1)
            data[f"{column}"] = (
                data[f"{column.replace('_ratio', '_auc')}"] / data["sum_auc"]
            )
        if normalize == "time":
            data = calc_ratio(data, column.replace("_ratio", ""))

    # Calculate tertiles per participant, over conditions
    if calc_tertile_over_condition:
        data["condition"] = data.groupby("participant")[column].transform(
            lambda x: pd.qcut(x, q=3, labels=["Low", "Medium", "High"])
        )

    # Set up plot
    fig, ax = plt.subplots(1, len(conditions), sharey=True, figsize=(8, 4))
    plt.setp(ax, ylim=(0.5, 1.1))
    ax[0].set_ylabel("P(response == 1)")

    for i, condition in enumerate(conditions):
        # Subset to condition
        data_subset = data[data.SAT == condition]
        if not calc_tertile_over_condition:
            data_subset["condition"] = data_subset.groupby("participant")[
                column
            ].transform(lambda x: pd.qcut(x, q=3, labels=["Low", "Medium", "High"]))
        # Calculate P(response == 1), per participant and per condition
        participant_ratios = (
            data_subset.groupby(["participant", "condition"], observed=True)
            .response.mean()
            .reset_index()
        )

        i_ax = ax[i]
        sns.violinplot(
            x="condition",
            y="response",
            data=participant_ratios,
            hue="condition",
            palette="mako_r",
            ax=i_ax,
            cut=0,
        )

        # Plot participant lines
        # for participant in participant_ratios["participant"].unique():
        #     participant_data = participant_ratios[
        #         participant_ratios["participant"] == participant
        #     ]
        #     i_ax.plot(
        #         participant_data["condition"],
        #         participant_data["response"],
        #         marker=".",
        #         color="black",
        #         alpha=0.3,
        #     )

        # i_ax.set_title(f"{condition.capitalize()} (n={len(data_subset)})")
        i_ax.set_title(f"{condition.capitalize()}")
        i_ax.set_xlabel("")
        col_words = column.split("_")
        fig.supxlabel(f"Average {col_words[0]} probability tertile")
        i_ax.set_ylabel(f"Probability of correct response")

        # t-test subsets
        low_data = participant_ratios[participant_ratios["condition"] == "Low"][
            "response"
        ]
        medium_data = participant_ratios[participant_ratios["condition"] == "Medium"][
            "response"
        ]
        high_data = participant_ratios[participant_ratios["condition"] == "High"][
            "response"
        ]

        # Perform pairwise t-tests
        ttest_low_medium = ttest_ind(low_data, medium_data)
        ttest_medium_high = ttest_ind(medium_data, high_data)
        ttest_low_high = ttest_ind(low_data, high_data)

        # Add significance annotations to the plot
        top = participant_ratios.response.max()
        add_significance_annotations(ttest_low_medium.pvalue, 0, 1, top + 0.01, i_ax)
        add_significance_annotations(ttest_medium_high.pvalue, 1, 2, top + 0.05, i_ax)
        add_significance_annotations(ttest_low_high.pvalue, 0, 2, top + 0.09, i_ax)
    plt.tight_layout()
    plt.savefig(f"../../img/tertile_split_{column}.svg", transparent=True)
    plt.show()


def display_trial(model, dataset, behaviour, idx: int, labels):
    data = dataset.__getitem__(idx)
    epoch, true, info = data[0], data[1], data[2][0]
    print(info)
    # plot_eeg(epoch)
    if behaviour is not None:
        p_trial_info = match_on_event_name(
            info["event_name"], behaviour, info["participant"], info["rt"]
        )
        print(p_trial_info)
        rt_ratio = p_trial_info["rt"].item() / get_masking_index(epoch)
        print(f"RT/masking index ratio: {rt_ratio}")
    plot_predictions_on_epoch(
        epoch,
        true,
        labels,
        window_size=0,
        model=model,
        smoothing=False,
        sequence=True,
        random_perm=True,
        save=False,
    )


def plot_distributions(data, process="confirmation"):
    set_seaborn_style()
    auc_columns = [
        column
        for column in data.columns
        if column.endswith("_auc") and not column.startswith("negative")
    ]
    data["sum_auc"] = data[auc_columns].sum(axis=1)
    # data['rt_x_norm'] = (data['rt_x'] - data['rt_x'].min()) / (data['rt_x'].max() - data['rt_x'].min())
    # Normalized operation certainty
    data["rt_samples"] = data["rt_x"] * (250)
    x_col = process + "_auc"
    data["ratio"] = data[x_col] / data["rt_samples"]
    bins = 75
    element = "poly"  # bars, step, poly
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
    sns.histplot(
        data,
        x="rt_samples",
        element=element,
        ax=ax[0],
        bins=bins,
        hue="SAT",
        palette=sns.color_palette()[1:],
        legend=False,
    )
    sns.histplot(
        data,
        x=x_col,
        element=element,
        ax=ax[1],
        bins=bins,
        hue="SAT",
        palette=sns.color_palette()[1:],
        legend=False,
    )
    sns.histplot(
        data,
        x="ratio",
        element=element,
        ax=ax[2],
        bins=bins,
        hue="SAT",
        palette=sns.color_palette()[1:],
    )
    # plt.legend()
    plt.ylim(0, 3000)
    ax[0].set_xlabel("RT (in samples)")
    ax[0].set_ylabel("Count (trials)")
    ax[1].set_xlabel(f"{process.capitalize()} AUC")
    ax[2].set_xlabel(f"Average {process} probability")
    plt.tight_layout()
    plt.savefig("../../img/dists.svg", transparent=True)
    plt.show()


def plot_loss(losses, labels):
    set_seaborn_style()
    losses = torch.stack(losses)
    df = pd.DataFrame(losses.detach().cpu().numpy()[:, 1:], columns=labels[1:])
    df["x"] = df.index  # Add x-axis column (index represents the time points)
    df_melted = df.melt(id_vars="x", var_name="Dimension", value_name="Value")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x="x", y="Value", hue="Dimension")
    plt.title("Tensor Visualization")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend(title="Dimension")
    plt.show()


def plot_cumulative(model, loader, labels):
    # SAT2
    torch.cuda.empty_cache()

    target_length = 100

    speed_pred = torch.zeros((target_length, len(labels)))
    speed_true = torch.zeros((target_length, len(labels)))
    accuracy_pred = torch.zeros((target_length, len(labels)))
    accuracy_true = torch.zeros((target_length, len(labels)))

    n_speed = 0
    n_accuracy = 0

    with torch.no_grad():
        for batch in loader:
            info = batch[2][0]

            # Predict on batch
            pred = model(batch[0].to(DEVICE))
            pred = torch.nn.Softmax(dim=2)(pred).to("cpu")
            true = batch[1]

            lengths = get_masking_indices(batch[0])

            for i in range(pred.shape[0]):
                is_speed = "speed" in info["event_name"][i]
                pred_epoch = pred[i, : lengths[i]].permute(1, 0).unsqueeze(0)
                true_epoch = true[i, : lengths[i]].permute(1, 0).unsqueeze(0)

                pred_normalized = torch.nn.functional.interpolate(
                    pred_epoch, size=target_length, mode="linear", align_corners=False
                )
                true_normalized = torch.nn.functional.interpolate(
                    true_epoch, size=target_length, mode="linear", align_corners=False
                )
                pred_normalized = pred_normalized / pred_normalized.sum(
                    dim=2, keepdim=True
                )
                true_normalized = true_normalized / true_normalized.sum(
                    dim=2, keepdim=True
                )

                # true_normalized *= lengths[i] / 100
                # pred_normalized *= lengths[i] / 100

                # Additionally normalize over sum to create a valid probability distribution

                pred_normalized = pred_normalized.squeeze().permute(1, 0)
                true_normalized = true_normalized.squeeze().permute(1, 0)

                if is_speed:
                    speed_pred += pred_normalized
                    speed_true += true_normalized
                    n_speed += 1
                else:
                    accuracy_pred += pred_normalized
                    accuracy_true += true_normalized
                    n_accuracy += 1

    speed_pred /= n_speed
    speed_true /= n_speed
    accuracy_pred /= n_accuracy
    accuracy_true /= n_accuracy

    # Set confirmation in speed to 0
    speed_true[:, 4] = 0

    set_seaborn_style()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    fig.supylabel("Cumulative probability")
    fig.supxlabel("Scaled time")

    ax[0].set_title(f"Accuracy")
    ax[1].set_title(f"Speed")
    # ax[0].set_title(f"Accuracy (n={n_accuracy})")
    # ax[1].set_title(f"Speed (n={n_speed})")
    for i, label in enumerate(labels):
        if i == 0:
            continue
        sns.lineplot(
            accuracy_true[:, i].cumsum(dim=0),
            ax=ax[0],
            label=label,
            color=sns.color_palette()[i],
            linestyle="--",
            alpha=0.5,
        )
        sns.lineplot(
            accuracy_pred[:, i].cumsum(dim=0),
            ax=ax[0],
            label=label,
            color=sns.color_palette()[i],
            linestyle="-",
        )

        sns.lineplot(
            speed_true[:, i].cumsum(dim=0),
            ax=ax[1],
            label=label,
            color=sns.color_palette()[i],
            linestyle="--",
            alpha=0.5,
        )
        sns.lineplot(
            speed_pred[:, i].cumsum(dim=0),
            ax=ax[1],
            label=label,
            color=sns.color_palette()[i],
            linestyle="-",
        )

    handles, ax_labels = ax[0].get_legend_handles_labels()
    label_legend = ax[0].legend(
        handles=handles[1::2],
        labels=ax_labels[1::2],
        title="Operation",
        loc="upper left",
    )

    # ax[1].legend(handles=handles, labels=ax_labels, title="Labels")
    custom_lines = [
        plt.Line2D([0], [0], color="black", linestyle="--"),  # Solid line
        plt.Line2D([0], [0], color="black", linestyle="-"),  # Dashed line
    ]

    line_legend = ax[0].legend(
        custom_lines, ["HMP", "S4"], loc="lower right", title="Source"
    )
    ax[0].add_artist(label_legend)

    plt.legend([], [], frameon=False)
    # ax[0].get_legend().remove()

    # ax[0].add_artist(line_legend)

    plt.tight_layout()
    plt.savefig("../../img/perf_plot_cumulative.svg", transparent=True)

    plt.show()


def plot_density(model, loader, labels):
    # SAT2
    torch.cuda.empty_cache()

    target_length = 100

    speed_pred = torch.zeros((target_length, len(labels)))
    speed_true = torch.zeros((target_length, len(labels)))
    accuracy_pred = torch.zeros((target_length, len(labels)))
    accuracy_true = torch.zeros((target_length, len(labels)))

    n_speed = 0
    n_accuracy = 0

    with torch.no_grad():
        for batch in loader:
            info = batch[2][0]

            # Predict on batch
            pred = model(batch[0].to(DEVICE))
            pred = torch.nn.Softmax(dim=2)(pred).to("cpu")
            true = batch[1]

            lengths = get_masking_indices(batch[0])

            for i in range(pred.shape[0]):
                is_speed = "speed" in info["event_name"][i]
                pred_epoch = pred[i, : lengths[i]].permute(1, 0).unsqueeze(0)
                true_epoch = true[i, : lengths[i]].permute(1, 0).unsqueeze(0)

                pred_normalized = torch.nn.functional.interpolate(
                    pred_epoch, size=target_length, mode="linear", align_corners=False
                )
                true_normalized = torch.nn.functional.interpolate(
                    true_epoch, size=target_length, mode="linear", align_corners=False
                )
                pred_normalized = pred_normalized / pred_normalized.sum(
                    dim=2, keepdim=True
                )
                true_normalized = true_normalized / true_normalized.sum(
                    dim=2, keepdim=True
                )

                # Additionally normalize over sum to create a valid probability distribution
                pred_normalized = pred_normalized.squeeze().permute(1, 0)
                true_normalized = true_normalized.squeeze().permute(1, 0)

                if is_speed:
                    speed_pred += pred_normalized
                    speed_true += true_normalized
                    n_speed += 1
                else:
                    accuracy_pred += pred_normalized
                    accuracy_true += true_normalized
                    n_accuracy += 1

    speed_pred /= n_speed
    speed_true /= n_speed
    accuracy_pred /= n_accuracy
    accuracy_true /= n_accuracy

    # Set confirmation in speed to 0
    speed_true[:, 3] = 0

    set_seaborn_style()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    fig.supylabel("Probability density")
    fig.supxlabel("Scaled time")

    ax[0].set_title(f"Accuracy")
    ax[1].set_title(f"Speed")
    # ax[0].set_title(f"Accuracy (n={n_accuracy})")
    # ax[1].set_title(f"Speed (n={n_speed})")
    for i, label in enumerate(labels):
        if i == 0:
            continue
        sns.lineplot(
            accuracy_true[:, i],
            ax=ax[0],
            label=label,
            color=sns.color_palette()[i],
            linestyle="--",
            alpha=0.5,
        )
        sns.lineplot(
            accuracy_pred[:, i],
            ax=ax[0],
            label=label,
            color=sns.color_palette()[i],
            linestyle="-",
        )

        sns.lineplot(
            speed_true[:, i],
            ax=ax[1],
            label=label,
            color=sns.color_palette()[i],
            linestyle="--",
            alpha=0.5,
        )
        sns.lineplot(
            speed_pred[:, i],
            ax=ax[1],
            label=label,
            color=sns.color_palette()[i],
            linestyle="-",
        )

    handles, ax_labels = ax[0].get_legend_handles_labels()
    label_legend = ax[0].legend(
        handles=handles[1::2],
        labels=ax_labels[1::2],
        title="Operation",
        loc="upper left",
    )
    plt.ylim((0, 0.2))
    # ax[1].legend(handles=handles, labels=ax_labels, title="Labels")
    custom_lines = [
        plt.Line2D([0], [0], color="black", linestyle="--"),  # Solid line
        plt.Line2D([0], [0], color="black", linestyle="-"),  # Dashed line
    ]

    line_legend = ax[1].legend(
        custom_lines, ["HMP", "S4"], loc="upper right", title="Source"
    )
    ax[0].add_artist(label_legend)

    plt.legend([], [], frameon=False)
    # ax[0].get_legend().remove()

    ax[1].add_artist(line_legend)

    plt.tight_layout()
    plt.savefig("../../img/perf_plot_density.svg", transparent=True)

    plt.show()


def plot_peak_timing(model, loader, labels, ax_ac, ax_sp, cue_var="condition", path=None, sample=True):
    if path is None:
        output = []
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in loader:
                info = batch[2][0]  # Contains RT

                pred = model(batch[0].to(DEVICE))
                pred = torch.nn.Softmax(dim=2)(pred).to("cpu")

                true = batch[1]

                lengths = get_masking_indices(batch[0])

                pred_peaks = pred[..., 1:].argmax(dim=1).float()
                true_peaks = true[..., 1:].argmax(dim=1).float()

                pred_peaks /= lengths.unsqueeze(1)
                true_peaks /= lengths.unsqueeze(1)
                data = {"condition": info[cue_var]}
                for i, label in enumerate(labels):
                    if i == 0:
                        continue
                    label_pred_peaks = pred_peaks[:, i - 1]
                    label_true_peaks = true_peaks[:, i - 1]
                    data[f"{label}_pred"] = label_pred_peaks
                    data[f"{label}_true"] = label_true_peaks
                output.append(data)
        df = pd.concat([pd.DataFrame(data) for data in output])
        df.to_csv("visu_peak.csv", index=False)
    else:
        df = pd.read_csv(path)
    ac_label = "accuracy" if cue_var == "condition" else "AC"
    sp_label = "speed" if cue_var == "condition" else "SP"

    for i_label, label in enumerate(labels):
        if label == "negative":
            continue
        # sns.kdeplot(
        #     data=df[df["condition"] == ac_label],
        #     x=f"{label}_pred",
        #     y=f"{label}_true",
        #     fill=True,       # Fills the contour
        #     alpha=0.3,       # Transparency so overlaps are visible
        #     color=sns.color_palette()[i_label - 1],       # Distinct color for each class
        #     label=label,         # For legend
        #     ax=ax_ac
        # )
        # sns.kdeplot(
        #     data=df[df["condition"] == sp_label],
        #     x=f"{label}_pred",
        #     y=f"{label}_true",
        #     fill=True,       # Fills the contour
        #     alpha=0.3,       # Transparency so overlaps are visible
        #     color=sns.color_palette()[i_label - 1],       # Distinct color for each class
        #     label=label,         # For legend
        #     ax=ax_sp
        # )
        # sns.histplot(
        #     data=df[df["condition"] == ac_label],
        #     x=f"{label}_pred",
        #     y=f"{label}_true",
        #     bins=30,      # Fills the contour
        #     alpha=0.7,       # Transparency so overlaps are visible
        #     color=sns.color_palette()[i_label - 1],       # Distinct color for each class
        #     label=label,         # For legend
        #     ax=ax_ac
        # )
        # sns.histplot(
        #     data=df[df["condition"] == sp_label],
        #     x=f"{label}_pred",
        #     y=f"{label}_true",
        #     bins=30,      # Fills the contour
        #     alpha=0.7,       # Transparency so overlaps are visible
        #     color=sns.color_palette()[i_label - 1],       # Distinct color for each class
        #     label=label,         # For legend
        #     ax=ax_sp
        # )
        # sns.regplot(
        #     data=df[df["condition"] == ac_label],
        #     x=f"{label}_pred",
        #     y=f"{label}_true",
        #     scatter=False,
        #     ax=ax_ac,
        #     color=sns.color_palette()[i_label - 1],
        #     # robust=True,
        #     truncate=True,
        #     # n_boot=5000,
        # )
        scatter_subset = df[df["condition"] == ac_label]
        if sample:
            scatter_subset = scatter_subset.sample(frac=0.1)
        sns.scatterplot(
            scatter_subset,
            x=f"{label}_pred",
            y=f"{label}_true",
            alpha=0.2,
            ax=ax_ac,
            color=sns.color_palette()[i_label - 1],
            linewidth=0
        )

        # sns.regplot(
        #     data=df[df["condition"] == sp_label],
        #     x=f"{label}_pred",
        #     y=f"{label}_true",
        #     scatter=False,
        #     ax=ax_sp,
        #     color=sns.color_palette()[i_label - 1],
        #     # robust=True,
        #     # n_boot=5000,
        # )
        scatter_subset = df[df["condition"] == sp_label].sample(frac=0.1)
        sns.scatterplot(
            scatter_subset,
            x=f"{label}_pred",
            y=f"{label}_true",
            alpha=0.2,
            ax=ax_sp,
            color=sns.color_palette()[i_label - 1],
            linewidth=0
        )


def plot_single_epoch(data, labels: list[str], model: torch.nn.Module, ax):
    # Pass in result of dataset.__getitem__(idx)
    epoch, true = data[0], data[1]
    rt_idx = get_masking_index(epoch)

    # Cut to RT and remove negative class
    true = true[:rt_idx, 1:]
    set_seaborn_style()

    pred = model(epoch.unsqueeze(0).to(DEVICE))
    pred = torch.nn.Softmax(dim=2)(pred).squeeze()
    # Necessary?
    # pred = pred / pred.sum(dim=2, keepdim=True)

    pred = pred.cpu().detach().numpy()
    pred = pred[:rt_idx, 1:]

    # Plot HMP
    for i in range(0, true.shape[1]):
        sns.lineplot(
            x=range(len(true[:, i])),
            y=true[:, i],
            ax=ax,
            color=sns.color_palette()[i],
            # label=labels[i + 1],
            legend=False,
            alpha=0.5,
            linestyle="--",
        )

    # Plot predictions
    for i in range(0, pred.shape[1]):
        sns.lineplot(
            x=range(len(pred[:, i])),
            y=pred[:, i],
            ax=ax,
            color=sns.color_palette()[i],
            label=labels[i + 1],
            legend=False,
        )


def plot_density_single(model, loader, labels, ax_ac, ax_sp):
    # SAT2
    torch.cuda.empty_cache()

    target_length = 100

    speed_pred = torch.zeros((target_length, len(labels)))
    speed_true = torch.zeros((target_length, len(labels)))
    accuracy_pred = torch.zeros((target_length, len(labels)))
    accuracy_true = torch.zeros((target_length, len(labels)))

    n_speed = 0
    n_accuracy = 0

    with torch.no_grad():
        for batch in loader:
            info = batch[2][0]

            # Predict on batch
            pred = model(batch[0].to(DEVICE))
            pred = torch.nn.Softmax(dim=2)(pred).to("cpu")
            true = batch[1]

            lengths = get_masking_indices(batch[0])

            for i in range(pred.shape[0]):
                is_speed = "speed" in info["event_name"][i]
                pred_epoch = pred[i, : lengths[i]].permute(1, 0).unsqueeze(0)
                true_epoch = true[i, : lengths[i]].permute(1, 0).unsqueeze(0)

                pred_normalized = torch.nn.functional.interpolate(
                    pred_epoch, size=target_length, mode="linear", align_corners=False
                )
                true_normalized = torch.nn.functional.interpolate(
                    true_epoch, size=target_length, mode="linear", align_corners=False
                )
                pred_normalized = pred_normalized / pred_normalized.sum(
                    dim=2, keepdim=True
                )
                true_normalized = true_normalized / true_normalized.sum(
                    dim=2, keepdim=True
                )

                # Additionally normalize over sum to create a valid probability distribution
                pred_normalized = pred_normalized.squeeze().permute(1, 0)
                true_normalized = true_normalized.squeeze().permute(1, 0)

                if is_speed:
                    speed_pred += pred_normalized
                    speed_true += true_normalized
                    n_speed += 1
                else:
                    accuracy_pred += pred_normalized
                    accuracy_true += true_normalized
                    n_accuracy += 1

    speed_pred /= n_speed
    speed_true /= n_speed
    accuracy_pred /= n_accuracy
    accuracy_true /= n_accuracy

    # Set confirmation in speed to 0
    speed_true[:, 3] = 0  # SAT2
    torch.cuda.empty_cache()

    target_length = 100

    speed_pred = torch.zeros((target_length, len(labels)))
    speed_true = torch.zeros((target_length, len(labels)))
    accuracy_pred = torch.zeros((target_length, len(labels)))
    accuracy_true = torch.zeros((target_length, len(labels)))

    n_speed = 0
    n_accuracy = 0

    with torch.no_grad():
        for batch in loader:
            info = batch[2][0]

            # Predict on batch
            pred = model(batch[0].to(DEVICE))
            pred = torch.nn.Softmax(dim=2)(pred).to("cpu")
            true = batch[1]

            lengths = get_masking_indices(batch[0])

            for i in range(pred.shape[0]):
                is_speed = "speed" in info["event_name"][i]
                pred_epoch = pred[i, : lengths[i]].permute(1, 0).unsqueeze(0)
                true_epoch = true[i, : lengths[i]].permute(1, 0).unsqueeze(0)

                pred_normalized = torch.nn.functional.interpolate(
                    pred_epoch, size=target_length, mode="linear", align_corners=False
                )
                true_normalized = torch.nn.functional.interpolate(
                    true_epoch, size=target_length, mode="linear", align_corners=False
                )
                pred_normalized = pred_normalized / pred_normalized.sum(
                    dim=2, keepdim=True
                )
                true_normalized = true_normalized / true_normalized.sum(
                    dim=2, keepdim=True
                )

                # Additionally normalize over sum to create a valid probability distribution
                pred_normalized = pred_normalized.squeeze().permute(1, 0)
                true_normalized = true_normalized.squeeze().permute(1, 0)

                if is_speed:
                    speed_pred += pred_normalized
                    speed_true += true_normalized
                    n_speed += 1
                else:
                    accuracy_pred += pred_normalized
                    accuracy_true += true_normalized
                    n_accuracy += 1

    speed_pred /= n_speed
    speed_true /= n_speed
    accuracy_pred /= n_accuracy
    accuracy_true /= n_accuracy

    # Set confirmation in speed to 0
    speed_true[:, 3] = 0

    for i, label in enumerate(labels):
        if i == 0:
            continue
        sns.lineplot(
            accuracy_true[:, i],
            ax=ax_ac,
            label=label,
            color=sns.color_palette()[i],
            linestyle="--",
            alpha=0.5,
            legend=False,
        )
        sns.lineplot(
            accuracy_pred[:, i],
            ax=ax_ac,
            label=label,
            color=sns.color_palette()[i],
            linestyle="-",
            legend=False,
        )

        sns.lineplot(
            speed_true[:, i],
            ax=ax_sp,
            label=label,
            color=sns.color_palette()[i],
            linestyle="--",
            alpha=0.5,
            legend=False,
        )
        sns.lineplot(
            speed_pred[:, i],
            ax=ax_sp,
            label=label,
            color=sns.color_palette()[i],
            linestyle="-",
            legend=False,
        )


def plot_tertile_split_single(
    data: pd.DataFrame,
    column: str,
    conditions: list[str],
    calc_tertile_over_condition: bool = False,
    normalize: str = "auc",
    axes: list = [],
    cue_var="SAT",
):
    pd.options.mode.chained_assignment = None
    data = data.copy()

    # Also do ratio here? Does not make sense since for non-confirmation operations we are certain that they exist?
    rt_col = "RT" if not "rt_x" in data else "rt_x"
    if column.endswith("_ratio"):
        if normalize == "auc":
            auc_columns = [
                column
                for column in data.columns
                if column.endswith("_auc") and not column.startswith("negative")
            ]
            data["sum_auc"] = data[auc_columns].sum(axis=1)
            data[f"{column}"] = (
                data[f"{column.replace('_ratio', '_auc')}"] / data["sum_auc"]
            )
        if normalize == "time":
            data = calc_ratio(data, column.replace("_ratio", ""), rt_col=rt_col)

    # Calculate tertiles per participant, over conditions
    if calc_tertile_over_condition:
        data["condition"] = data.groupby("participant")[column].transform(
            lambda x: pd.qcut(x, q=3, labels=["Low", "Medium", "High"])
        )

    for i, condition in enumerate(conditions):
        data_subset = data[data[cue_var] == condition]
        if not calc_tertile_over_condition:
            data_subset["condition"] = data_subset.groupby("participant")[
                column
            ].transform(lambda x: pd.qcut(x, q=3, labels=["Low", "Medium", "High"]))
        # Calculate P(response == 1), per participant and per condition
        participant_ratios = (
            data_subset.groupby(["participant", "condition"], observed=True)
            .response.mean()
            .reset_index()
        )
        sns.violinplot(
            x="condition",
            y="response",
            data=participant_ratios,
            # hue="condition",
            # palette="mako_r",
            color=sns.color_palette()[4],
            ax=axes[i],
            cut=0,
            label="Correct response",
            legend=False,
        )

def plot_emg_sequence_combined(data, ax):
    # Combine EMG_sequence groups
    group_mapping = {
        "IR": "1",
        "CR": "1",
        "ICR": "2",
        "CIR": "2",
        "CCR": "2",
        "IIR": "2",
    }
    data["EMG_group"] = data["EMG_sequence"].map(group_mapping)

    # Calculate ratio
    filtered_data = calc_ratio(data, "confirmation", normalize=True)

    # Create the violin plot
    sns.violinplot(
        x="EMG_group",
        y="confirmation_ratio",
        data=filtered_data,
        order=["1", "2"],
        cut=0,
        ax=ax,
    )

def plot_emg_tertile_split(data, axes, conditions, cue_var='SAT'):
    # Combine EMG_sequence groups
    group_mapping = {
        "IR": 0,
        "CR": 0,
        "ICR": 1,
        "CIR": 1,
        "CCR": 1,
        "IIR": 1,
    }
    data["EMG_group"] = data["EMG_sequence"].map(group_mapping)

    # Calculate ratio
    data = calc_ratio(data, "confirmation", normalize=True)
    column = "confirmation" + "_ratio"
    for i, condition in enumerate(conditions):
        data_subset = data[data[cue_var] == condition]
        data_subset["condition"] = data_subset.groupby("participant")[
            column
        ].transform(lambda x: pd.qcut(x, q=3, labels=["Low", "Medium", "High"]))
        # Calculate P(response == 1), per participant and per condition
        participant_ratios = (
            data_subset.groupby(["participant", "condition"], observed=True)
            .EMG_group.mean()
            .reset_index()
        )
        sns.violinplot(
            x="condition",
            y="EMG_group",
            data=participant_ratios,
            # hue="condition",
            # palette="mako_r",
            color=sns.color_palette()[5],
            ax=axes[i],
            cut=0,
            linewidth=1,
            label="Second EMG Event",
            legend=False,
        )