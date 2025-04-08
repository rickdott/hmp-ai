from hmpai.pytorch.utilities import DEVICE
from torch.utils.data import DataLoader
import torch
from hmpai.utilities import (
    calc_ratio,
    get_masking_index,
    get_masking_indices,
)
import seaborn as sns
import pandas as pd
from hmpai.utilities import set_seaborn_style


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

        batch_aucs = torch.sum(pred, dim=1)
        batch_aucs = batch_aucs.cpu().detach()

        # Append AUCs
        for i, label in enumerate(labels):
            data[label + "_auc"] = batch_aucs[:, i]
        data_list.append(pd.DataFrame(data))
    data = pd.concat(data_list)
    return data


def plot_peak_timing(
    model, loader, labels, ax_ac, ax_sp, cue_var="condition", path=None, sample=True
):
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
        df.to_csv("files/visu_peak.csv", index=False)
    else:
        df = pd.read_csv(path)
    ac_label = "accuracy" if cue_var == "condition" else "AC"
    sp_label = "speed" if cue_var == "condition" else "SP"

    for i_label, label in enumerate(labels):
        if label == "negative":
            continue

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
            linewidth=0,
        )

        scatter_subset = df[df["condition"] == sp_label].sample(frac=0.1)
        sns.scatterplot(
            scatter_subset,
            x=f"{label}_pred",
            y=f"{label}_true",
            alpha=0.2,
            ax=ax_sp,
            color=sns.color_palette()[i_label - 1],
            linewidth=0,
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


def plot_emg_tertile_split(data, axes, conditions, cue_var="SAT"):
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
        data_subset["condition"] = data_subset.groupby("participant")[column].transform(
            lambda x: pd.qcut(x, q=3, labels=["Low", "Medium", "High"])
        )
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
            color=sns.color_palette()[5],
            ax=axes[i],
            cut=0,
            linewidth=1,
            label="Second EMG Event",
            legend=False,
        )
