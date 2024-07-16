import logging
import os
import random

import cv2
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        dataset,
        size: int = 256,
        device: str = "cpu",
    ):
        self.device = device
        self.dataset = dataset
        self.class_names = sorted(set(self.dataset["name"]))
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}

        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
        elif isinstance(image, torch.Tensor):
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        return image.float().to(self.device)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        class_name = sample["name"]
        label = self.class_mapping[class_name]
        img_tensor = self.preprocess(img)

        return img_tensor, label, class_name

    @property
    def num_classes(self):
        return len(self.class_names)


class PreprocessedCatDataset(Dataset):
    def __init__(
        self,
        dataset,
        size: int = 256,
        device: str = "cpu",
    ):
        self.device = device
        self.dataset = dataset
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.3,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
        elif isinstance(image, torch.Tensor):
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        return image.float().to(self.device)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        img_tensor = self.preprocess(img)
        label = 2  # Fixed label for all cat images
        return img_tensor, label, "cat"  # Returning "cat" as class_name for consistency

    @property
    def num_classes(self):
        return 1


def visualize_dataset_samples(dataset, output_dir, num_samples=10, shuffle=False):
    os.makedirs(output_dir, exist_ok=True)

    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in tqdm(
        range(min(num_samples, len(dataset))), desc="Saving visualized samples"
    ):
        img_tensor, label, class_name = dataset[indices[i]]

        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        line_type = 2

        text_size = cv2.getTextSize(class_name, font, font_scale, line_type)[0]
        text_x = 10
        text_y = text_size[1] + 10

        cv2.rectangle(
            img_bgr,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            img_bgr,
            class_name,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            line_type,
        )

        output_path = os.path.join(output_dir, f"sample_{i:03d}_{class_name}.jpg")
        cv2.imwrite(output_path, img_bgr)

    logging.info(f"Saved {num_samples} visualized samples to {output_dir}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load the dataset
    from datasets import load_dataset

    logging.info("Loading the Smithsonian Butterflies dataset...")
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

    # Visualize some samples from the dataset
    logging.info("Visualizing samples from the dataset...")
    visualize_dataset_samples(
        preprocessed_dataset, "visualized_samples", num_samples=20, shuffle=True
    )
