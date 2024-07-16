import logging
import os
import random

import cv2
import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from tqdm import tqdm

from dataset import (
    PreprocessedDataset,
    PreprocessedCatDataset,
    visualize_dataset_samples,
)


def save_image_stack(
    dataset, output_path, num_images=25, grid_size=(5, 5), image_size=(128, 128)
):
    """
    Save a stack of random images from the dataset as a single image, without class name labels.

    Args:
    dataset (PreprocessedDataset): The dataset to sample images from.
    output_path (str): Path to save the output image.
    num_images (int): Number of images to include in the stack. Default is 25.
    grid_size (tuple): The grid layout for the images. Default is (5, 5).
    image_size (tuple): The size of each individual image in the grid. Default is (128, 128).
    """
    assert (
        num_images == grid_size[0] * grid_size[1]
    ), "num_images must equal grid_size[0] * grid_size[1]"

    # Create a blank canvas for the grid
    canvas = np.zeros(
        (grid_size[0] * image_size[0], grid_size[1] * image_size[1], 3), dtype=np.uint8
    )

    # Randomly sample indices
    indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(indices):
        img_tensor, _, _ = dataset[idx]

        # Convert tensor to numpy array and denormalize
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)

        # Resize the image
        img_resized = cv2.resize(img_np, image_size)

        # Calculate position in the grid
        row = i // grid_size[1]
        col = i % grid_size[1]

        # Place the image in the canvas
        canvas[
            row * image_size[0] : (row + 1) * image_size[0],
            col * image_size[1] : (col + 1) * image_size[1],
        ] = img_resized

    # Save the image
    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    logging.info(f"Saved image stack to {output_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load the dataset
    from datasets import load_dataset

    logging.info("Loading the dataset...")
    original_dataset = load_dataset("huggan/cats", split="train")

    # Create the PreprocessedDataset
    logging.info("Creating PreprocessedDataset...")
    preprocessed_dataset = PreprocessedCatDataset(
        dataset=original_dataset,
        size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Test the loaded dataset
    logging.info(f"Loaded dataset size: {len(preprocessed_dataset)}")
    sample_image, sample_label, sample_class_name = preprocessed_dataset[0]
    logging.info(
        f"Sample image shape: {sample_image.shape}, Sample label: {sample_label}, Sample class name: {sample_class_name}"
    )

    # Save a stack of random images
    logging.info("Saving a stack of random images...")
    save_image_stack(
        preprocessed_dataset,
        "image_stack.jpg",
        num_images=5,
        grid_size=(1, 5),
        image_size=(128, 128),
    )
