import base64
import io
import json
import logging
import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def save_images_with_overlay(dataset, output_dir):
    """
    Save images from the dataset with class name overlays,
    maintaining the original image resolution.

    Args:
    dataset (list): List of tuples (image, label, class_name).
    output_dir (str): Directory to save the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (image, label, class_name) in enumerate(dataset):
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Get image dimensions
        height, width = image_np.shape[:2]

        # Calculate font scale based on image size
        font_scale = min(width, height) / 500.0
        thickness = max(1, int(font_scale * 2))

        # Add class name overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(class_name, font, font_scale, thickness)[0]
        text_x = 10
        text_y = text_size[1] + 10

        cv2.putText(
            image_np,
            class_name,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_np,
            class_name,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        # Save the image
        output_path = os.path.join(output_dir, f"image_{idx:03d}_{class_name}.png")
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(dataset)} images to {output_dir}")


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        data_dir,
        size: int = 256,
        device: str = "cpu",
        reindex_classes: bool = True,
    ):
        self.data_dir = data_dir
        self.device = device
        self.samples = []
        self.class_names = {}
        self.reindex_classes = reindex_classes
        self.class_mapping = {}
        self.load_metadata()

        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def load_metadata(self):
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if self.reindex_classes:
            new_label = 0
            for class_name, class_info in metadata["classes"].items():
                original_label = class_info["label"]
                self.class_mapping[original_label] = new_label
                self.class_names[new_label] = class_name
                new_label += 1
        else:
            for class_name, class_info in metadata["classes"].items():
                self.class_names[class_info["label"]] = class_name

        for class_name, class_info in metadata["classes"].items():
            class_dir = os.path.join(
                self.data_dir, f"{class_name}_{class_info['num_samples']}_samples"
            )
            for shard_idx in range(class_info["num_shards"]):
                shard_path = os.path.join(class_dir, f"shard_{shard_idx:05d}.json")
                with open(shard_path, "r") as f:
                    self.samples.extend(json.load(f))

    def __len__(self):
        return len(self.samples)

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
        elif isinstance(image, torch.Tensor):
            # If it's already a tensor, ensure it has 3 channels
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        return image.float().to(self.device)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(
            io.BytesIO(base64.b64decode(sample["image"]["base64"]))
        ).convert("RGB")

        original_label = sample["label"]

        if self.reindex_classes:
            label = self.class_mapping[original_label]
        else:
            label = original_label

        class_name = self.class_names[label]
        img_tensor = self.preprocess(img)

        return img_tensor, label, class_name

    @property
    def num_classes(self):
        return len(self.class_names)


def save_imagenet_subset_shards(
    dataset,
    output_dir,
    selected_class_numbers=None,
    samples_per_class=None,
    elements_per_shard=500,
):
    """
    Save a subset of the ImageNet-1k dataset as shards, organized by class.
    Allows selection of specific classes by their numbers.

    Args:
    dataset: The original dataset.
    output_dir (str): Directory to save the shards.
    selected_class_numbers (list): List of class numbers to include. If None, include all classes.
    samples_per_class (int): Number of samples per class. If None, include all samples.
    elements_per_shard (int): Maximum number of elements in each shard.
    """
    logging.info(
        f"Starting save_imagenet_subset_shards with parameters: selected_class_numbers={selected_class_numbers}, samples_per_class={samples_per_class}, elements_per_shard={elements_per_shard}"
    )
    os.makedirs(output_dir, exist_ok=True)

    labels = dataset.features["label"].names
    selected_classes = (
        set(selected_class_numbers)
        if selected_class_numbers
        else set(range(len(labels)))
    )
    class_samples = defaultdict(list)

    total_classes = len(selected_classes)
    total_samples = len(dataset)

    logging.info(f"Total classes in dataset: {len(labels)}")
    logging.info(f"Total samples in dataset: {total_samples}")
    logging.info(f"Selected classes: {selected_classes}")

    # Main progress bar for processing samples
    main_pbar = tqdm(total=total_samples, desc="Collecting samples")

    for sample in dataset:
        try:
            label = sample["label"]
            class_name = labels[label]

            # Check if this class should be included
            if label in selected_classes:
                if (
                    samples_per_class is None
                    or len(class_samples[label]) < samples_per_class
                ):
                    # Check if 'image' is in the sample and if it's a PIL Image
                    if "image" not in sample:
                        raise KeyError("'image' key not found in sample")
                    if not isinstance(sample["image"], Image.Image):
                        raise ValueError(
                            f"Unexpected image data format: {type(sample['image'])}"
                        )

                    # Convert PIL Image to bytes, then to base64 string for storage
                    img_byte_arr = io.BytesIO()
                    sample["image"].save(img_byte_arr, format="JPEG")
                    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode(
                        "utf-8"
                    )

                    # Store the sample with base64 encoded image
                    class_samples[label].append(
                        {"image": {"base64": img_base64}, "label": sample["label"]}
                    )

            main_pbar.update(1)
            main_pbar.set_postfix(
                {
                    "Classes": f"{len(class_samples)}/{total_classes}",
                    "Samples": f"{sum(len(samples) for samples in class_samples.values())}",
                }
            )

            # Check if we've collected all the samples we need
            if samples_per_class is not None:
                if all(
                    label in class_samples
                    and len(class_samples[label]) >= samples_per_class
                    for label in selected_classes
                ):
                    logging.info(
                        "Reached target number of samples for all selected classes"
                    )
                    break

        except Exception as e:
            logging.error(f"Error processing sample for class {class_name}: {str(e)}")
            logging.error(f"Sample structure: {sample.keys()}")
            if "image" in sample:
                logging.error(f"Image type: {type(sample['image'])}")

    main_pbar.close()
    logging.info(
        f"Finished collecting samples. Selected {len(class_samples)} classes with a total of {sum(len(samples) for samples in class_samples.values())} samples."
    )

    # Save shards for each class
    metadata = {"classes": {}}

    class_pbar = tqdm(class_samples.keys(), desc="Processing classes")
    for label in class_pbar:
        class_name = labels[label]
        class_pbar.set_postfix({"Current class": class_name})

        samples = class_samples[label]
        class_dir = os.path.join(output_dir, f"{class_name}_{len(samples)}_samples")
        os.makedirs(class_dir, exist_ok=True)

        num_shards = math.ceil(len(samples) / elements_per_shard)
        logging.info(f"Saving {num_shards} shards for class {class_name}")

        # Progress bar for saving shards of current class
        shard_pbar = tqdm(
            total=num_shards, desc=f"Saving shards for {class_name}", leave=False
        )

        for shard_idx in range(num_shards):
            start_idx = shard_idx * elements_per_shard
            end_idx = min((shard_idx + 1) * elements_per_shard, len(samples))
            shard_samples = samples[start_idx:end_idx]

            shard_path = os.path.join(class_dir, f"shard_{shard_idx:05d}.json")
            with open(shard_path, "w") as f:
                json.dump(shard_samples, f)
            shard_pbar.update(1)

        shard_pbar.close()

        metadata["classes"][class_name] = {
            "label": label,
            "num_samples": len(samples),
            "num_shards": num_shards,
        }
        logging.info(f"Finished saving shards for class {class_name}")

    class_pbar.close()

    # Save metadata
    metadata["total_samples"] = sum(
        class_info["num_samples"] for class_info in metadata["classes"].values()
    )
    metadata["total_classes"] = len(metadata["classes"])
    metadata["elements_per_shard"] = elements_per_shard

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    logging.info(
        f"Saved metadata. Total samples: {metadata['total_samples']}, Total classes: {metadata['total_classes']}"
    )
    return metadata


def visualize_dataset_samples(dataset, output_dir, num_samples=10, shuffle=False):
    """
    Save the first n images from a PreprocessedDataset with class name overlays.

    Args:
    dataset (PreprocessedDataset): The dataset to visualize.
    output_dir (str): Directory to save the visualized images.
    num_samples (int): Number of samples to visualize. Default is 10.
    shuffle (bool): Whether to shuffle the dataset before selecting samples. Default is False.
    """
    os.makedirs(output_dir, exist_ok=True)

    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in tqdm(
        range(min(num_samples, len(dataset))), desc="Saving visualized samples"
    ):
        img_tensor, label, class_name = dataset[indices[i]]

        # Convert tensor to numpy array and change from CxHxW to HxWxC
        img_np = img_tensor.permute(1, 2, 0).numpy()

        # Denormalize the image
        img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)

        # Convert from RGB to BGR (OpenCV uses BGR)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Add class name overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White color
        line_type = 2

        # Get text size
        text_size = cv2.getTextSize(class_name, font, font_scale, line_type)[0]

        # Position the text at the top-left corner with some padding
        text_x = 10
        text_y = text_size[1] + 10

        # Draw black rectangle as background for text
        cv2.rectangle(
            img_bgr,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1,
        )

        # Put text on the image
        cv2.putText(
            img_bgr,
            class_name,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            line_type,
        )

        # Save the image
        output_path = os.path.join(output_dir, f"sample_{i:03d}_{class_name}.jpg")
        cv2.imwrite(output_path, img_bgr)

    logging.info(f"Saved {num_samples} visualized samples to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Load the original dataset
    logging.info("Loading original ImageNet dataset...")
    original_dataset = load_dataset("imagenet-1k", split="train", use_auth_token=True)

    dog_list = np.arange(151, 269).tolist()

    # Save a subset as shards
    # logging.info("Saving subset as shards...")
    # metadata = save_imagenet_subset_shards(
    #     original_dataset,
    #     output_dir="imagenet_subset_shards",
    #     selected_class_numbers=dog_list,
    #     samples_per_class=None,
    #     elements_per_shard=200,
    # )

    # Load the entire dataset using PreprocessedDataset
    logging.info("Loading entire dataset using PreprocessedDataset...")
    preprocessed_dataset = PreprocessedDataset(
        "imagenet_subset_shards", size=256, device="cuda"
    )

    # Test the loaded dataset
    logging.info(f"Loaded dataset size: {len(preprocessed_dataset)}")
    sample_image, sample_label, sample_class_name = preprocessed_dataset[0]
    logging.info(
        f"Sample image shape: {sample_image.shape}, Sample label: {sample_label}, Sample class name: {sample_class_name}"
    )

    # Visualize some samples from the dataset
    logging.info("Visualizing samples from the dataset...")
    visualize_dataset_samples(
        preprocessed_dataset, "visualized_samples", num_samples=20, shuffle=True
    )
