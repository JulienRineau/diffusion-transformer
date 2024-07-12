import os
import random

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image


def load_imagenet_subset(num_samples=100, split="train"):
    """
    Load a subset of the ImageNet-1k dataset from Hugging Face,
    keeping the original image resolution.

    Args:
    num_samples (int): Number of samples to load. Default is 100.
    split (str): Dataset split to use. Default is "train".

    Returns:
    list: A list of tuples, each containing (image, label, class_name).
    """
    dataset = load_dataset("imagenet-1k", split=split, use_auth_token=True)
    labels = dataset.features["label"].names

    annotated_images = []
    for sample in dataset.take(num_samples):
        image = sample["image"]
        label = sample["label"]
        class_name = labels[label]
        annotated_images.append((image, label, class_name))

    return annotated_images


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
