from collections import defaultdict, Counter
from mne.viz import plot_topomap
from mne import Info
import numpy as np
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
from hmpai.data import SAT1_STAGES_ACCURACY, preprocess
import captum
from hmpai.pytorch.generators import SAT1Dataset
from hmpai.pytorch.utilities import DEVICE
from torch.utils.data import DataLoader
import torch
from hmpai.utilities import MASKING_VALUE
from tqdm.notebook import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy.stats import ttest_rel
from typing import List, Optional
import json


def add_attribution(
    dataset: xr.Dataset, analyzer: captum.attr.Attribution, model: torch.nn.Module
) -> xr.Dataset:
    """
    Analyzes the given dataset using the provided attribution method and model, and adds the analysis
    results to the dataset.

    Args:
        dataset (xr.Dataset): The dataset to analyze.
        analyzer (captum.attr.Attribution): The attribution method to use for analysis.
        model (torch.nn.Module): The model to use for analysis.

    Returns:
        xr.Dataset: The analyzed dataset with the analysis results added.
    """
    test_set = preprocess(dataset)
    test_dataset = SAT1Dataset(test_set, do_preprocessing=False)
    test_set = test_set.assign(
        analysis=(("index", "samples", "channels"), np.zeros_like(test_set.data))
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, pin_memory=True
    )

    # Batch-wise analyzing of data and adding analysis to the dataset
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # print(f"Batch: {i + 1}/{batches}")
        batch_data = batch[0].to(DEVICE)
        baselines = torch.clone(batch_data)
        mask = baselines != MASKING_VALUE
        baselines[mask] = 0
        baselines = baselines.to(DEVICE)
        target = torch.argmax(model(batch_data), dim=1)
        batch_analysis = analyzer.attribute(
            batch_data,
            baselines=baselines,
            n_steps=50,
            method="riemann_trapezoid",
            # Change to batch[1] if true values should be used instead of model predictions
            target=target,
            internal_batch_size=128,
        )
        test_set.analysis[i * len(batch[1]) : (i + 1) * len(batch[1])] = torch.squeeze(
            batch_analysis.cpu()
        )

    return test_set


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


def set_seaborn_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("tab10")
    # Dutch field
    # sns.set_palette(sns.color_palette(["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]))


def plot_predictions_on_epoch(
    epoch: torch.Tensor,
    true: torch.Tensor,
    labels: list[str],
    window_size: int,
    model: torch.nn.Module,
    smoothing: bool = False,
    sequence: bool = False,
    random_perm: bool = False,
):
    def smooth_predictions(predictions, window_size):
        smoothed = np.copy(predictions)
        for i in range(window_size // 2, len(predictions) - window_size // 2):
            smoothed[i] = np.mean(
                predictions[i - window_size // 2 : i + window_size // 2 + 1], axis=0
            )
        return smoothed

    set_seaborn_style()
    empty = np.zeros((epoch.size()[0], len(labels)))
    rt_idx = torch.nonzero(torch.eq(epoch[:, 0], MASKING_VALUE))[0, 0].item()
    # Normalize probability labels
    true = true[:rt_idx, 1:]
    true = true / true.sum()
    if not sequence:
        slices = get_padded_slices(epoch, window_size, include_start_end=False)
        stacked = torch.stack(slices).to(DEVICE)
        pred = model(stacked)
        pred = torch.nn.Softmax(dim=1)(pred)
    else:
        if random_perm:
            perm = torch.randperm(rt_idx)
            epoch[:rt_idx] = epoch[perm]
            true[:rt_idx] = true[perm]
        pred = model(epoch.unsqueeze(0).to(DEVICE))
        pred = torch.nn.Softmax(dim=2)(pred)
    pred = pred.cpu().detach().numpy()
    if smoothing:
        pred = smooth_predictions(pred, window_size)

    if not sequence:
        for i, prediction in enumerate(pred):
            empty[i, :] = prediction
            # empty[i + window_size // 2, :] = prediction
    else:
        empty = pred.squeeze()
    # print(true)
    fig, ax = plt.subplots()
    # for i in range(0, 5):
    #     mask = torch.eq(true, i)
    #     indices = torch.nonzero(mask, as_tuple=True)
    #     if indices[0].numel() == 0:
    #         continue
    #     ax.barh(
    #         0.9,
    #         torch.sum(mask).item(),
    #         left=indices[0][0],
    #         align="center",
    #         alpha=0.5,
    #         height=0.05,
    #         color=sns.color_palette()[i],
    #     )
    for i in range(0, empty.shape[1]):
        sns.lineplot(
            x=range(len(empty[:, i])),
            y=empty[:, i],
            ax=ax,
            color=sns.color_palette()[i],
            label=labels[i],
        )
    ax2 = ax.twinx()
    for i in range(0, true.shape[1]):
        sns.lineplot(
            x=range(len(true[:rt_idx, i])),
            y=true[:rt_idx, i],
            ax=ax2,
            color = sns.color_palette()[i + 4],
            label=labels[i]
        )
    # sns.lineplot(empty[:, 1:], ax=ax)
    # label=SAT_CLASSES_ACCURACY[1:]
    ax.legend()
    plt.xlim(0, rt_idx + (window_size // 2))
    # plt.ylim(0, 1.0)
    plt.xlabel("Samples")
    plt.ylabel("Softmax probability")
    plt.show()

def plot_stage_predictions(epoch: torch.Tensor, labels: list[str], window_size: int, model: torch.nn.Module): 
    set_seaborn_style()
    empty = np.zeros((epoch.size()[0], len(labels)))
    rt_idx = torch.nonzero(torch.eq(epoch[:, 0], MASKING_VALUE))[0, 0].item()

    slices = get_padded_slices(epoch, window_size, include_start_end=False)
    stacked = torch.stack(slices).to(DEVICE)
    # Pad to the size the model is used to
    stacked = torch.nn.functional.pad(stacked, (0, 0, 0, 100 - window_size), value=MASKING_VALUE)
    try:
        pred = model(stacked)
    except Exception:
        print('Oopsie')
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


def get_padded_slices(epoch: torch.Tensor, window_size: int, include_start_end: bool = True):
    # Get index where epoch ends
    rt_idx = torch.nonzero(torch.eq(epoch[:, 0], MASKING_VALUE))[0, 0].item()
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
        data = {info_key: batch[2][info_key] for info_key in info_to_keep}
        pred = model(batch[0].to(DEVICE))
        pred = torch.nn.Softmax(dim=2)(pred)
        pred = pred.cpu().detach()
        batch_aucs = torch.sum(pred, dim=1)
        predicted_labels = pred.argmax(dim=2) # (batch_size, seq_len)
        for i in range(predicted_labels.shape[0]):
            trial_predictions = predicted_labels[i,:]
            unique_counts = torch.unique(trial_predictions, return_counts=True)
            unique = list(unique_counts[0])
            counts = list(unique_counts[1])
            for j, label in enumerate(labels):
                if label + "_pred_samples" not in data:
                    data[label + "_pred_samples"] = []
                try:
                    data[label + "_pred_samples"].append(counts[unique.index(j)].item())
                except ValueError:
                    data[label + "_pred_samples"].append(0)
        for i in range(batch[1].shape[0]):
            trial_labels = batch[1][i, :]
            unique_counts = torch.unique(trial_labels, return_counts=True)
            unique = list(unique_counts[0])
            counts = list(unique_counts[1])
            for j, label in enumerate(labels):
                if label + "_true_samples" not in data:
                    data[label + "_true_samples"] = []
                try:
                    data[label + "_true_samples"].append(counts[unique.index(j)].item())
                except ValueError:
                    data[label + "_true_samples"].append(0)
        for i, label in enumerate(labels):
            data[label + "_auc"] = batch_aucs[:, i]
        data_list.append(pd.DataFrame(data))
    data = pd.concat(data_list)
    return data


def plot_median_split_error_rate(data: pd.DataFrame, operation: str):
    set_seaborn_style()
    data["ratio"] = (data[f"{operation}_auc"] / 100) / data["rt_x"]
    median = data["ratio"].median()
    data["above_median"] = data["ratio"] >= median
    plt.figure(figsize=(4, 6))
    plot = sns.barplot(data=data, x="SAT", y="response", hue="above_median", legend=True)
    plt.ylabel("Proportion of correct responses")
    plt.xlabel(f"Below or above median of {operation} AUC/RT\n(Median: {median:.2f})")
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