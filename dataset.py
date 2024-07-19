import os
import logging
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from vae import StableDiffusionVAE


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        dataset,
        size: int = 256,
        device: str = "cpu",
        shard_dir: str = "dataset_shards",
        num_shards: int = 10,
    ):
        self.device = device
        self.dataset = dataset
        self.shard_dir = shard_dir
        self.num_shards = num_shards
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.vae = StableDiffusionVAE().to(device)
        self.vae.vae.eval()

        self.shards = self.load_or_create_shards()
        self.shard_sizes = [len(shard) for shard in self.shards]
        self.cumulative_sizes = [0] + list(np.cumsum(self.shard_sizes))

    def load_or_create_shards(self):
        if self.shards_exist():
            return self.load_shards()
        else:
            return self.create_shards()

    def shards_exist(self):
        return all(os.path.exists(self.shard_path(i)) for i in range(self.num_shards))

    def shard_path(self, shard_idx):
        return os.path.join(self.shard_dir, f"shard_{shard_idx}.pt")

    def load_shards(self):
        logging.info("Loading existing shards...")
        return [torch.load(self.shard_path(i)) for i in tqdm(range(self.num_shards))]

    def create_shards(self):
        logging.info("Creating new shards...")
        os.makedirs(self.shard_dir, exist_ok=True)
        shards = [[] for _ in range(self.num_shards)]

        for idx in tqdm(range(len(self.dataset)), desc="Processing dataset"):
            sample = self.dataset[idx]
            img = sample["image"]
            img_tensor = self.preprocess(img)

            with torch.no_grad():
                latent = self.vae.encode(img_tensor.unsqueeze(0).to(self.device))

            shard_idx = idx % self.num_shards
            shards[shard_idx].append(
                (img_tensor.cpu(), latent.cpu(), 0, "cat")
            )  # 0 is the label for cats

        for i, shard in enumerate(shards):
            torch.save(shard, self.shard_path(i))

        return shards

    def __len__(self):
        return sum(self.shard_sizes)

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
        elif isinstance(image, torch.Tensor):
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        return image.float()

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Find which shard the index belongs to
        shard_idx = np.searchsorted(self.cumulative_sizes, idx, side="right") - 1

        # Calculate the index within the shard
        local_idx = idx - self.cumulative_sizes[shard_idx]

        img_tensor, latent_tensor, label, class_name = self.shards[shard_idx][local_idx]
        return (
            img_tensor.to(self.device),
            latent_tensor.squeeze(0).to(self.device),
            label,
            class_name,
        )

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
        img_tensor, latent_tensor, label, class_name = dataset[indices[i]]

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load the dataset
    logging.info("Loading the cat dataset...")
    original_dataset = load_dataset("huggan/cats", split="train")

    # Create the PreprocessedDataset
    logging.info("Creating PreprocessedDataset...")
    preprocessed_dataset = PreprocessedDataset(
        dataset=original_dataset,
        size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        shard_dir="cat_dataset_shards",
        num_shards=10,
    )

    # Test the loaded dataset
    logging.info(f"Loaded dataset size: {len(preprocessed_dataset)}")
    sample_img_tensor, sample_latent_tensor, sample_label, sample_class_name = (
        preprocessed_dataset[0]
    )
    logging.info(
        f"Sample image shape: {sample_img_tensor.shape}, "
        f"Sample latent shape: {sample_latent_tensor.shape}, "
        f"Sample label: {sample_label}, "
        f"Sample class name: {sample_class_name}"
    )

    # Visualize some samples from the dataset
    logging.info("Visualizing samples from the dataset...")
    visualize_dataset_samples(
        preprocessed_dataset, "visualized_samples", num_samples=5, shuffle=True
    )
