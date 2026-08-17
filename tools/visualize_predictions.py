import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from patchsgg.config import load_config
from patchsgg.data.factory import build_dataset
from patchsgg.model import build_vocab


PRED_FILE = "outputs/location_free_pix2sg_paper/10_predictions.json"
CONFIG = "patchsgg/configs/location_free_paper.yaml"
OUT_DIR = Path("outputs/location_free_pix2sg_paper/vis")


def find_image_path(ds, image_id):
    for idx, meta in enumerate(ds.image_meta):
        if int(meta["image_id"]) == int(image_id):
            return ds._image_path(idx)

    raise ValueError(f"image_id {image_id} not found")


def draw_graph(ax, title, graph_text):
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    # limit number of relations for readability
    relations = graph_text.split(";")[:30]

    text = "\n".join(relations)

    ax.text(
        0.0,
        1.0,
        text,
        verticalalignment="top",
        fontsize=20,
        wrap=True,
    )


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(CONFIG, [])

    vocab = build_vocab(cfg)

    ds = build_dataset(cfg, "val", vocab)

    with open(PRED_FILE, "r") as f:
        predictions = json.load(f)

    for item in predictions:

        image_id = item["image_id"]

        image_path = find_image_path(ds, image_id)

        image = Image.open(image_path).convert("RGB")

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(24, 12)
        )

        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title(f"Image ID: {image_id}")

        draw_graph(
            axes[1],
            "Ground Truth",
            item["gt"]
        )

        draw_graph(
            axes[2],
            "Prediction",
            item["pred"]
        )

        out = OUT_DIR / f"{image_id}.png"

        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

        print(f"saved {out}")


if __name__ == "__main__":
    main()