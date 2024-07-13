import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def load_imagenet_subset(
    num_classes=5, samples_per_class=20, split="train", streaming=False
):
    """
    Load a subset of the ImageNet-1k dataset from Hugging Face,
    selecting a specified number of classes and samples per class.

    Args:
    num_classes (int): Number of classes to include. Default is 5.
    samples_per_class (int): Number of samples per class. Default is 20.
    split (str): Dataset split to use. Default is "train".
    streaming (bool): Whether to use streaming mode. Default is True.

    Returns:
    list: A list of tuples, each containing (image, label, class_name).
    list: List of selected class labels.
    """
    print(f"Loading ImageNet-1k dataset (split: {split}, streaming: {streaming})")
    dataset = load_dataset(
        "imagenet-1k", split=split, use_auth_token=True, streaming=streaming
    )

    labels = dataset.features["label"].names
    selected_classes = set()
    class_counts = defaultdict(int)
    annotated_images = []

    progress_bar = tqdm(desc=f"Processing samples for {num_classes} classes")

    for sample in dataset:
        image = sample["image"]
        label = sample["label"]
        class_name = labels[label]

        if len(selected_classes) < num_classes or label in selected_classes:
            if class_counts[label] < samples_per_class:
                annotated_images.append((image, label, class_name))
                class_counts[label] += 1
                selected_classes.add(label)
                progress_bar.update(1)
                progress_bar.set_postfix({"Current class": class_name})

        if len(selected_classes) == num_classes and all(
            count == samples_per_class for count in class_counts.values()
        ):
            break

    progress_bar.close()
    print(
        f"Loaded {len(annotated_images)} samples from {len(selected_classes)} classes"
    )

    return annotated_images, list(selected_classes)


class PreprocessedDataset(Dataset):
    def __init__(self, original_dataset, size: int = 256, device: str = "cpu"):
        self.original_dataset = original_dataset
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.original_dataset)

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
        elif isinstance(image, torch.Tensor):
            # If it's already a tensor, ensure it has 3 channels
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        return image.float()

    def __getitem__(self, idx):
        img = self.original_dataset[idx]["image"]
        label = self.original_dataset[idx]["label"]
        class_name = self.original_dataset.features["label"].names[label]
        img_tensor = self.preprocess(img)
        return img_tensor, label, class_name


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


# Example usage
if __name__ == "__main__":
    # Load the dataset
    dataset = load_imagenet_subset(num_samples=10)

    # Save images with overlays
    save_images_with_overlay(dataset, "output_images")
