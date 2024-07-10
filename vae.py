import torch
from transformers import AutoFeatureExtractor, AutoTokenizer
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


class StableDiffusionVAE:
    def __init__(self, model_path="CompVis/stable-diffusion-v1-4", subfolder="vae"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder=subfolder).to(
            self.device
        )
        self.vae.eval()

        # Load the feature extractor for proper image preprocessing
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    def preprocess(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        return image.to(self.device)

    def encode(self, image):
        image = self.preprocess(image)
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def decode(self, latent):
        latent = latent / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        return image


def process_and_save_image(input_path, output_path):
    # Load the image
    input_image = Image.open(input_path).convert("RGB")

    # Initialize VAE and process image
    vae = StableDiffusionVAE()
    latent = vae.encode(input_image)
    reconstructed = vae.decode(latent)

    # Convert tensors to numpy arrays for visualization
    input_np = np.array(input_image)
    latent_np = latent.squeeze().cpu().numpy()
    latent_np = (latent_np - latent_np.min()) / (latent_np.max() - latent_np.min())
    latent_np = (latent_np * 255).astype(np.uint8)
    reconstructed_np = (
        reconstructed.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    ) * 255

    # Create the visualization
    h, w = input_np.shape[:2]
    vis_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis_image[:, :w] = input_np
    vis_image[:, w : 2 * w] = cv2.resize(latent_np[0], (w, h))[:, :, np.newaxis].repeat(
        3, axis=2
    )
    vis_image[:, 2 * w :] = cv2.resize(reconstructed_np.astype(np.uint8), (w, h))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_image, "Input", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis_image, "Latent", (w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        vis_image,
        "Reconstructed",
        (2 * w + 10, 30),
        font,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    input_path = "path/to/your/input/image.jpg"
    output_path = "path/to/your/output/visualization.jpg"
    process_and_save_image(input_path, output_path)
