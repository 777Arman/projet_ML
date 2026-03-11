"""
extract_features.py

PROVIDED TO STUDENTS — do not modify.

Reads images from data/images/, generates sliding window region proposals,
labels each proposal using IoU against ground truth boxes from data/regions.csv,
extracts ResNet-50 features for each proposal, and saves:

    data/features_train.npz
    data/features_val.npz

Each .npz file contains:
    features   : float32 array, shape (N, 2048)
    labels     : int64 array,   shape (N,)
    boxes      : float32 array, shape (N, 4)  — [x1, y1, x2, y2]
    image_ids  : int64 array,   shape (N,)

Label encoding:
    0 = background
    1 = person
    2 = car
    3 = truck

Usage:
    python extract_features.py --data_dir ./data --batch_size 64
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import torchvision.models as models
import torchvision.transforms as T

# label encoding
LABEL_MAP = {"background": 0, "person": 1, "car": 2, "truck": 3}

# ResNet input size
RESNET_INPUT_SIZE = 224

# IoU thresholds for region labeling
IOU_POSITIVE_THRESHOLD = 0.5   # IoU >= 0.5  -> foreground
IOU_IGNORE_THRESHOLD   = 0.3   # IoU in [0.3, 0.5) -> discard, IoU < 0.3 -> background

# sliding window configuration
SW_SCALES        = [64, 128, 256]    # box sizes in pixels
SW_ASPECT_RATIOS = [0.5, 1.0, 2.0]  # height/width ratios
SW_STRIDE        = 32               # stride in pixels

# how many background regions to keep per foreground region
# keeping all backgrounds would use too much RAM and is not necessary
BACKGROUND_RATIO = 3

# save to disk every this many images to avoid running out of RAM
SAVE_EVERY = 200


# build ResNet-50 without its classification head so it outputs 2048-d vectors
def build_feature_extractor(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)
    return model


# standard ImageNet normalization that ResNet expects
RESNET_TRANSFORM = T.Compose([
    T.Resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def sliding_window_proposals(img_w, img_h):
    """Generate candidate boxes by sliding windows of different sizes across the image."""
    boxes = []
    for scale in SW_SCALES:
        for ratio in SW_ASPECT_RATIOS:
            box_w = int(scale)
            box_h = int(scale * ratio)
            if box_w > img_w or box_h > img_h:
                continue
            for y in range(0, img_h - box_h + 1, SW_STRIDE):
                for x in range(0, img_w - box_w + 1, SW_STRIDE):
                    x2 = min(x + box_w, img_w)
                    y2 = min(y + box_h, img_h)
                    boxes.append([x, y, x2, y2])
    return np.array(boxes, dtype=np.float32)


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU between every pair of boxes. Returns shape (N, M)."""
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def label_proposals(proposals, gt_boxes, gt_labels):
    """
    Assign a label to each proposed region based on how much it overlaps
    with ground truth boxes.

    IoU >= 0.5  -> class label of the overlapping ground truth box (foreground)
    IoU <  0.3  -> background (label 0)
    IoU in between -> discard (label -1)
    """
    if len(gt_boxes) == 0:
        return np.zeros(len(proposals), dtype=np.int64)

    iou     = compute_iou_matrix(proposals, gt_boxes)
    max_iou = iou.max(axis=1)
    best_gt = iou.argmax(axis=1)

    labels = np.full(len(proposals), -1, dtype=np.int64)
    labels[max_iou <  IOU_IGNORE_THRESHOLD]   = 0
    labels[max_iou >= IOU_POSITIVE_THRESHOLD] = gt_labels[
        best_gt[max_iou >= IOU_POSITIVE_THRESHOLD]
    ]
    return labels


def subsample_background(labels, rng):
    """
    Keep all foreground regions but only a fraction of background regions.
    This prevents background from overwhelming RAM and the classifier.
    The ratio is controlled by BACKGROUND_RATIO.
    """
    fg_indices = np.where(labels > 0)[0]
    bg_indices = np.where(labels == 0)[0]

    n_bg_keep = min(len(bg_indices), len(fg_indices) * BACKGROUND_RATIO)
    if n_bg_keep < len(bg_indices):
        bg_indices = rng.choice(bg_indices, size=n_bg_keep, replace=False)

    keep = np.concatenate([fg_indices, bg_indices])
    keep.sort()
    return keep


@torch.no_grad()
def extract_batch(crops, model, device):
    """Pass a batch of PIL image crops through ResNet and return 2048-d feature vectors."""
    tensors = torch.stack([RESNET_TRANSFORM(c) for c in crops]).to(device)
    feats   = model(tensors)                     # (B, 2048, 1, 1)
    feats   = feats.squeeze(-1).squeeze(-1)      # (B, 2048)
    return feats.cpu().numpy()


def append_to_npz(out_file, features, labels, boxes, image_ids):
    """
    Append a chunk of results to the output file.
    If the file already exists, load it and concatenate before saving.
    This keeps RAM usage low by flushing to disk regularly.
    """
    if out_file.exists():
        existing = np.load(out_file)
        features  = np.concatenate([existing["features"],  features],  axis=0)
        labels    = np.concatenate([existing["labels"],    labels],    axis=0)
        boxes     = np.concatenate([existing["boxes"],     boxes],     axis=0)
        image_ids = np.concatenate([existing["image_ids"], image_ids], axis=0)

    np.savez_compressed(
        out_file,
        features  = features,
        labels    = labels,
        boxes     = boxes,
        image_ids = image_ids,
    )


def process_split(split_name, image_list, images_dir, gt_by_image,
                  model, device, batch_size, out_file, rng):
    """
    Run the full pipeline for a list of images and save results incrementally
    to avoid running out of RAM.
    """
    # remove any previous partial output for this split
    if out_file.exists():
        out_file.unlink()

    chunk_features  = []
    chunk_labels    = []
    chunk_boxes     = []
    chunk_image_ids = []

    total = len(image_list)

    for idx, (image_id, file_name) in enumerate(image_list):

        if idx % 50 == 0:
            print(f"  [{split_name}] {idx}/{total} images processed...")

        img_path = images_dir / file_name
        if not img_path.exists():
            print(f"  Warning: {img_path} not found, skipping.")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        proposals = sliding_window_proposals(img_w, img_h)
        if len(proposals) == 0:
            continue

        gt = gt_by_image.get(image_id, [])
        if len(gt) > 0:
            gt_boxes  = np.array([[g[0], g[1], g[2], g[3]] for g in gt], dtype=np.float32)
            gt_labels = np.array([g[4] for g in gt], dtype=np.int64)
        else:
            gt_boxes  = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros(0, dtype=np.int64)

        labels = label_proposals(proposals, gt_boxes, gt_labels)

        # discard ambiguous regions
        keep_mask = labels != -1
        proposals = proposals[keep_mask]
        labels    = labels[keep_mask]

        if len(proposals) == 0:
            continue

        # subsample background to keep memory manageable
        keep = subsample_background(labels, rng)
        proposals = proposals[keep]
        labels    = labels[keep]

        # extract features in batches
        batch_crops  = []
        batch_boxes  = []
        batch_labels = []

        for i, (box, label) in enumerate(zip(proposals, labels)):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop = img.crop((x1, y1, x2, y2))
            batch_crops.append(crop)
            batch_boxes.append(box)
            batch_labels.append(label)

            if len(batch_crops) == batch_size or i == len(proposals) - 1:
                feats = extract_batch(batch_crops, model, device)
                chunk_features.append(feats)
                chunk_boxes.extend(batch_boxes)
                chunk_labels.extend(batch_labels)
                chunk_image_ids.extend([image_id] * len(batch_crops))
                batch_crops  = []
                batch_boxes  = []
                batch_labels = []

        # flush to disk every SAVE_EVERY images to free RAM
        if (idx + 1) % SAVE_EVERY == 0:
            print(f"  [{split_name}] Saving checkpoint at image {idx + 1}...")
            append_to_npz(
                out_file,
                np.vstack(chunk_features).astype(np.float32),
                np.array(chunk_labels,    dtype=np.int64),
                np.array(chunk_boxes,     dtype=np.float32),
                np.array(chunk_image_ids, dtype=np.int64),
            )
            chunk_features  = []
            chunk_labels    = []
            chunk_boxes     = []
            chunk_image_ids = []

    # save whatever is left in the last chunk
    if chunk_features:
        print(f"  [{split_name}] Saving final chunk...")
        append_to_npz(
            out_file,
            np.vstack(chunk_features).astype(np.float32),
            np.array(chunk_labels,    dtype=np.int64),
            np.array(chunk_boxes,     dtype=np.float32),
            np.array(chunk_image_ids, dtype=np.int64),
        )

    # print final distribution
    final = np.load(out_file)
    labels_all  = final["labels"]
    label_names = {v: k for k, v in LABEL_MAP.items()}
    print(f"\n  {split_name} region distribution:")
    unique, counts = np.unique(labels_all, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {label_names.get(u, str(u)):<12}: {c:>8} ({100*c/len(labels_all):.1f}%)")
    print(f"  Saved to {out_file}. Total regions: {len(labels_all)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data",
                        help="Directory containing images/, regions.csv, selected_images.csv")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of region crops per ResNet forward pass")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for background subsampling")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    images_dir = data_dir / "images"
    rng        = np.random.default_rng(args.seed)

    print("Loading metadata...")
    selected = pd.read_csv(data_dir / "selected_images.csv")
    regions  = pd.read_csv(data_dir / "regions.csv")

    # build ground truth lookup: image_id -> list of (x1, y1, x2, y2, label)
    gt_by_image = defaultdict(list)
    for _, row in regions.iterrows():
        label = LABEL_MAP.get(row["class_label"], -1)
        if label == -1:
            continue
        gt_by_image[int(row["image_id"])].append((
            float(row["x1"]), float(row["y1"]),
            float(row["x2"]), float(row["y2"]),
            label
        ))
    all_imgs = [(int(r["image_id"]), r["file_name"]) for _, r in selected.iterrows()]
    rng_split = np.random.default_rng(args.seed)
    rng_split.shuffle(all_imgs)
    split_idx  = int(len(all_imgs) * 0.75)
    train_imgs = all_imgs[:split_idx]
    val_imgs   = all_imgs[split_idx:]

    print(f"Train images: {len(train_imgs)} | Val images: {len(val_imgs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading ResNet-50...")
    model = build_feature_extractor(device)

    for split_name, image_list, out_file in [
        ("train", train_imgs, data_dir / "features_train.npz"),
        ("val",   val_imgs,   data_dir / "features_val.npz"),
    ]:
        print(f"\nProcessing {split_name} split ({len(image_list)} images)...")
        process_split(
            split_name, image_list, images_dir,
            gt_by_image, model, device, args.batch_size,
            out_file, rng
        )

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
