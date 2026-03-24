import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates


CSV_PATH = "learning.csv"


def load_learning_data(path):
    data = pd.read_csv(path, header=None)
    if data.shape[1] < 2:
        raise ValueError("learning.csv must contain at least one feature column and one winner column.")

    feature_count = data.shape[1] - 1
    feature_columns = [f"feature_{idx}" for idx in range(1, feature_count + 1)]
    columns = feature_columns + ["winner"]
    data.columns = columns

    data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=feature_columns + ["winner"])

    return data, feature_columns


def normalize_features(dataframe, feature_columns):
    normalized = dataframe.copy()
    for column in feature_columns:
        col_min = normalized[column].min()
        col_max = normalized[column].max()
        if col_max == col_min:
            normalized[column] = 0.5
        else:
            normalized[column] = (normalized[column] - col_min) / (col_max - col_min)
    return normalized


def main():
    data, feature_columns = load_learning_data(CSV_PATH)
    if data.empty:
        raise ValueError("No valid rows found in learning.csv after cleaning.")

    plot_df = normalize_features(data, feature_columns)

    winner_counts = plot_df["winner"].value_counts().sort_index()

    print("Winner counts:")
    for winner, count in winner_counts.items():
        print(f"  {winner}: {count}")

    fig, ax = plt.subplots(figsize=(14, 8))

    unique_winners = sorted(plot_df["winner"].unique())
    cmap = plt.get_cmap("tab10", len(unique_winners))
    color_map = {winner: cmap(idx) for idx, winner in enumerate(unique_winners)}
    colors = [color_map[winner] for winner in unique_winners]

    parallel_coordinates(
        plot_df[feature_columns + ["winner"]],
        class_column="winner",
        color=colors,
        alpha=0.08,
        linewidth=0.9,
        ax=ax,
    )

    total_rows = len(plot_df)
    summary_text = "\n".join([f"{winner}: {count}" for winner, count in winner_counts.items()])

    ax.set_xlabel("Parameters")
    ax.set_ylabel("Normalized Value (0 to 1)")
    ax.set_title(f"Winner Space Across All Parameters | Total={total_rows}")
    ax.grid(alpha=0.25, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {label: handle for handle, label in zip(handles, labels)}
    legend_labels = [f"{winner} (n={winner_counts[winner]})" for winner in unique_winners]
    legend_handles = [label_to_handle[winner] for winner in unique_winners if winner in label_to_handle]
    ax.legend(legend_handles, legend_labels, loc="upper left", frameon=True)

    fig.text(
        0.02,
        0.98,
        "Winner counts\n" + summary_text,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()