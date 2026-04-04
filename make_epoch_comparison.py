from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path("samples/20260320")
OUTPUT = ROOT / "epoch_comparison_basecand_vs_smaat_unet.png"

MODELS = [
    {
        "label": "Conv LSTM",
        "pred_dir": ROOT / "base_network_cand" / "420j6ajc" / "preds",
        "epochs": [(0, "Epoch 1"), (19, "Epoch 20")],
    },
    {
        "label": "SmaAt-UNet",
        "pred_dir": ROOT / "smaat_unet" / "lvw79g7j" / "preds",
        "epochs": [(0, "Epoch 1"), (19, "Epoch 20")],
    },
]

LEADS = [1, 2, 3, 4, 5]


def build_rows():
    rows = []
    for model in MODELS:
        for epoch_idx, epoch_label in model["epochs"]:
            image_paths = [
                model["pred_dir"] / f"epoch{epoch_idx}_b0_T{lead}.png"
                for lead in LEADS
            ]
            missing = [str(path) for path in image_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing input images: {missing}")

            rows.append(
                {
                    "row_label": f'{model["label"]}\n{epoch_label}',
                    "paths": image_paths,
                }
            )
    return rows


def main():
    rows = build_rows()

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "figure.titlesize": 24,
        }
    )

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=len(LEADS),
        figsize=(20, 16),
        dpi=300,
        constrained_layout=False,
    )

    fig.patch.set_facecolor("white")

    for col_idx, lead in enumerate(LEADS):
        axes[0, col_idx].set_title(f"Lead T{lead}", fontweight="bold", pad=14)

    for row_idx, row in enumerate(rows):
        for col_idx, image_path in enumerate(row["paths"]):
            ax = axes[row_idx, col_idx]
            ax.imshow(mpimg.imread(image_path))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_edgecolor("#444444")

        axes[row_idx, 0].set_ylabel(
            row["row_label"],
            rotation=0,
            labelpad=90,
            va="center",
            ha="right",
            fontweight="bold",
            fontsize=17,
        )

    # fig.suptitle(
    #     "Prediction Comparison by Model and Training Epoch",
    #     y=0.98,
    #     fontweight="bold",
    # )
    # fig.text(
    #     0.5,
    #     0.945,
    #     "Rows compare Epoch 1 and Epoch 20 outputs for each model; columns show forecast leads T1 to T5.",
    #     ha="center",
    #     va="center",
    #     fontsize=15,
    # )

    plt.subplots_adjust(left=0.18, right=0.99, top=0.91, bottom=0.04, wspace=0.04, hspace=0.09)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(OUTPUT)


if __name__ == "__main__":
    main()
