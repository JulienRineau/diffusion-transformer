import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm


class StableDiffusionVAE:
    def __init__(self, model_path="stabilityai/sd-vae-ft-mse"):
        self.vae = AutoencoderKL.from_pretrained(model_path)
        self.vae.eval()
        self.scaling_factor = 0.18215

    def to(self, device):
        self.vae = self.vae.to(device)
        return self

    def encode(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.scaling_factor
        return latent

    def decode(self, latent):
        latent = latent / self.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        return image


def process_and_save_image(vae, input_image, output_path, class_name):
    # Ensure the input image is a 4D tensor (batch, channels, height, width)
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)

    # Encode and decode the input image
    latent = vae.encode(input_image)
    reconstructed = vae.decode(latent)

    # Convert tensors to numpy arrays
    input_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
    latent_np = latent.float().squeeze().cpu().numpy()
    latent_np = np.mean(latent_np, axis=0)  # Average across channels
    reconstructed_np = reconstructed.squeeze().permute(1, 2, 0).cpu().numpy()

    # Clip values to [0, 1] range
    input_np = np.clip(input_np, 0, 1)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    print("Loading the cat dataset...")
    dataset = load_dataset("huggan/cats", split="train")

    # Initialize VAE
    vae = StableDiffusionVAE().to(device)

    # Define transform
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    output_dir = "output_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Process a subset of images (e.g., 20)
    num_samples = 20
    for idx in tqdm(range(num_samples), desc="Processing images"):
        sample = dataset[idx]
        image = sample["image"]

        # Apply transform
        image_tensor = transform(image).unsqueeze(0).to(device)

        output_path = os.path.join(output_dir, f"visualization_cat_{idx:03d}.png")
        process_and_save_image(vae, image_tensor, output_path, "Cat")

    print(f"Processed {num_samples} images. Visualizations saved in {output_dir}")
