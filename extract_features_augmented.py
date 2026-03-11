"""
extract_features_augmented.py

Task 2a — Region-level augmentation for foreground classes.

Modified version of extract_features.py that applies data augmentation
to foreground regions (person, car, truck) before extracting ResNet features.
This creates additional training samples to mitigate class imbalance.

Background regions are NOT augmented.

Produces:
    coco_filtered/features_train_augmented.npz

Usage:
    python extract_features_augmented.py --data_dir ./coco_filtered --batch_size 64 --augment_factor 2
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
IOU_POSITIVE_THRESHOLD = 0.5
IOU_IGNORE_THRESHOLD   = 0.3

# sliding window configuration
SW_SCALES        = [64, 128, 256]
SW_ASPECT_RATIOS = [0.5, 1.0, 2.0]
SW_STRIDE        = 32

# background ratio
BACKGROUND_RATIO = 3

# save to disk every this many images
SAVE_EVERY = 200


def build_feature_extractor(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)
    return model


# Standard ResNet normalization (no augmentation)
RESNET_TRANSFORM = T.Compose([
    T.Resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# Augmentation pipeline for foreground regions
AUGMENT_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    T.RandomRotation(degrees=15),
    T.Resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def sliding_window_proposals(img_w, img_h):
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
    fg_indices = np.where(labels > 0)[0]
    bg_indices = np.where(labels == 0)[0]

    n_bg_keep = min(len(bg_indices), len(fg_indices) * BACKGROUND_RATIO)
    if n_bg_keep < len(bg_indices):
        bg_indices = rng.choice(bg_indices, size=n_bg_keep, replace=False)

    keep = np.concatenate([fg_indices, bg_indices])
    keep.sort()
    return keep


@torch.no_grad()
def extract_batch(crops, model, device, transform=RESNET_TRANSFORM):
    """Pass a batch of PIL image crops through ResNet with specified transform."""
    tensors = torch.stack([transform(c) for c in crops]).to(device)
    feats   = model(tensors)
    feats   = feats.squeeze(-1).squeeze(-1)
    return feats.cpu().numpy()


def append_to_npz(out_file, features, labels, boxes, image_ids):
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


def process_split_augmented(split_name, image_list, images_dir, gt_by_image,
                            model, device, batch_size, out_file, rng, augment_factor):
    """
    Process images with augmentation for foreground regions.
    
    For each foreground region (person, car, truck):
    - Extract features from the original crop (1x)
    - Extract features from augmented versions (augment_factor - 1 times)
    
    Background regions are not augmented.
    """
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

        keep_mask = labels != -1
        proposals = proposals[keep_mask]
        labels    = labels[keep_mask]

        if len(proposals) == 0:
            continue

        keep = subsample_background(labels, rng)
        proposals = proposals[keep]
        labels    = labels[keep]

        # Extract features with augmentation for foreground
        batch_crops  = []
        batch_boxes  = []
        batch_labels = []
        batch_transforms = []

        for i, (box, label) in enumerate(zip(proposals, labels)):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop = img.crop((x1, y1, x2, y2))
            
            # Always add the original crop
            batch_crops.append(crop)
            batch_boxes.append(box)
            batch_labels.append(label)
            batch_transforms.append(RESNET_TRANSFORM)
            
            # If foreground (label > 0), add augmented versions
            if label > 0:
                for _ in range(augment_factor - 1):
                    batch_crops.append(crop)
                    batch_boxes.append(box)
                    batch_labels.append(label)
                    batch_transforms.append(AUGMENT_TRANSFORM)

            # Process batch when full or at the end
            if len(batch_crops) >= batch_size or i == len(proposals) - 1:
                # Group by transform type for efficient processing
                standard_idx = [j for j, t in enumerate(batch_transforms) if t == RESNET_TRANSFORM]
                augment_idx  = [j for j, t in enumerate(batch_transforms) if t == AUGMENT_TRANSFORM]
                
                all_feats = np.zeros((len(batch_crops), 2048), dtype=np.float32)
                
                if standard_idx:
                    standard_crops = [batch_crops[j] for j in standard_idx]
                    standard_feats = extract_batch(standard_crops, model, device, RESNET_TRANSFORM)
                    all_feats[standard_idx] = standard_feats
                
                if augment_idx:
                    augment_crops = [batch_crops[j] for j in augment_idx]
                    augment_feats = extract_batch(augment_crops, model, device, AUGMENT_TRANSFORM)
                    all_feats[augment_idx] = augment_feats
                
                chunk_features.append(all_feats)
                chunk_boxes.extend(batch_boxes)
                chunk_labels.extend(batch_labels)
                chunk_image_ids.extend([image_id] * len(batch_crops))
                
                batch_crops  = []
                batch_boxes  = []
                batch_labels = []
                batch_transforms = []

        # Save checkpoint periodically
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

    # Save final chunk
    if chunk_features:
        print(f"  [{split_name}] Saving final chunk...")
        append_to_npz(
            out_file,
            np.vstack(chunk_features).astype(np.float32),
            np.array(chunk_labels,    dtype=np.int64),
            np.array(chunk_boxes,     dtype=np.float32),
            np.array(chunk_image_ids, dtype=np.int64),
        )

    # Print final distribution
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
    parser.add_argument("--data_dir",   default="./coco_filtered")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--augment_factor", type=int, default=2,
                        help="Number of versions (original + augmented) per foreground region")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    images_dir = data_dir / "images"
    rng        = np.random.default_rng(args.seed)

    print("Loading metadata...")
    selected = pd.read_csv(data_dir / "selected_images.csv")
    regions  = pd.read_csv(data_dir / "regions.csv")

    gt_by_image = defaultdict(list)
    for _, row in regions.iterrows():
        label = LABEL_MAP.get(row["class_label"], -1)
        if label == -1 or label == 0:
            continue
        gt_by_image[int(row["image_id"])].append((
            float(row["x1"]), float(row["y1"]),
            float(row["x2"]), float(row["y2"]),
            label
        ))

    train_imgs = [(int(r["image_id"]), r["file_name"]) 
                  for _, r in selected.iterrows() if r["split"] == "train"]

    print(f"Train images: {len(train_imgs)}")
    print(f"Augmentation factor: {args.augment_factor}x for foreground regions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading ResNet-50...")
    model = build_feature_extractor(device)

    out_file = data_dir / "features_train_augmented.npz"
    print(f"\nProcessing train split with augmentation...")
    process_split_augmented(
        "train", train_imgs, images_dir,
        gt_by_image, model, device, args.batch_size,
        out_file, rng, args.augment_factor
    )

    print("\nAugmented feature extraction complete.")
    print(f"Output saved to: {out_file}")


if __name__ == "__main__":
    main()
