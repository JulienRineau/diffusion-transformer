import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

from dataset import load_imagenet_subset


class StableDiffusionVAE:
    def __init__(self, model_path="stabilityai/sd-vae-ft-mse"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained(model_path).to(self.device)
        self.vae.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            image = self.transform(image)
        return image.to(self.device).float()  # Ensure float32

    def encode(self, image):
        # Remove the unsqueeze operation as the input is already batched
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def decode(self, latent):
        latent = latent / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        return image


def process_and_save_image(vae, input_image, output_path, class_name):
    # Preprocess and encode the input image
    input_tensor = vae.preprocess(input_image)
    latent = vae.encode(input_image)
    reconstructed = vae.decode(latent)

    # Print shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Prepare input image
    input_np = np.array(input_image.resize((256, 256)))

    # Prepare latent representation
    latent_np = latent.float().squeeze().cpu().numpy()
    latent_np = np.mean(latent_np, axis=0)  # Average across channels

    # Prepare reconstructed image
    reconstructed_np = (
        reconstructed.float().squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    )
    reconstructed_np = (reconstructed_np * 255).astype(np.uint8)

    # Create the visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"VAE Visualization for {class_name}")

    # Plot input image
    ax1.imshow(input_np)
    ax1.set_title(f"Input\n{input_np.shape}")
    ax1.axis("off")

    # Plot latent representation
    latent_img = ax2.imshow(latent_np, cmap="viridis")
    ax2.set_title(f"Latent\n{latent_np.shape}")
    ax2.axis("off")
    plt.colorbar(latent_img, ax=ax2, fraction=0.046, pad=0.04)

    # Plot reconstructed image
    ax3.imshow(reconstructed_np)
    ax3.set_title(f"Reconstructed\n{reconstructed_np.shape}")
    ax3.axis("off")

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    dataset = load_imagenet_subset(num_samples=10)
    output_dir = "output_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize VAE once
    vae = StableDiffusionVAE()

    for idx, (image, label, class_name) in enumerate(dataset):
        output_path = os.path.join(
            output_dir, f"visualization_{idx:03d}_{class_name}.png"
        )
        process_and_save_image(vae, image, output_path, class_name)
