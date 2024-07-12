import multiprocessing
import os

import pytorch_lightning as pl
import torch
from diffusers import DDPMScheduler
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import wandb
from dataset import load_imagenet_subset
from dit import DiT, DiTConfig
from vae import StableDiffusionVAE

# Set the start method for multiprocessing
multiprocessing.set_start_method("spawn", force=True)


class PreprocessedDataset(Dataset):
    def __init__(self, original_dataset, vae):
        self.original_dataset = original_dataset
        self.vae = vae

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label, class_name = self.original_dataset[idx]
        img_tensor = self.vae.preprocess(img)
        return img_tensor, label, class_name


class DiTLightning(pl.LightningModule):
    def __init__(self, dit_config, vae_model_path):
        super().__init__()
        self.save_hyperparameters()
        self.net = DiT(dit_config)
        self.vae = StableDiffusionVAE(vae_model_path)
        self.loss_fn = nn.MSELoss()
        self.num_train_timesteps = 1000
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )

    def forward(self, x, t, class_labels):
        return self.net(x, t, class_labels)

    def training_step(self, batch, batch_idx):
        images, class_labels, _ = batch

        # Encode images using VAE
        with torch.no_grad():
            latents = self.vae.encode(images)

        # Sample noise to add to the latents
        noise = torch.randn_like(latents)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.num_train_timesteps, (latents.shape[0],), device=self.device
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.net(noisy_latents, timesteps, class_labels)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


def filter_dataset(dataset, num_classes=5, samples_per_class=100):
    # Get unique classes
    all_classes = set(label for _, label, _ in dataset)
    selected_classes = list(all_classes)[:num_classes]

    # Filter dataset
    filtered_indices = [
        i for i, (_, label, _) in enumerate(dataset) if label in selected_classes
    ]

    # Limit samples per class
    class_counts = {label: 0 for label in selected_classes}
    final_indices = []
    for idx in tqdm(filtered_indices, desc="Filtering dataset"):
        _, label, _ = dataset[idx]
        if class_counts[label] < samples_per_class:
            final_indices.append(idx)
            class_counts[label] += 1
        if all(count == samples_per_class for count in class_counts.values()):
            break

    return Subset(dataset, final_indices), selected_classes


def save_class_images(dataset, selected_classes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_classes = set()

    for img, label, class_name in tqdm(dataset, desc="Saving class images"):
        if label in selected_classes and label not in saved_classes:
            img_path = os.path.join(output_dir, f"{class_name}_{label}.png")
            img.save(img_path)  # Save PIL Image directly
            saved_classes.add(label)
            print(f"Saved image for class {class_name} (label {label}) to {img_path}")

        if len(saved_classes) == len(selected_classes):
            break


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="dit", log_model="all")

    # Load ImageNet subset
    full_dataset = load_imagenet_subset(
        num_samples=10000
    )  # Load more samples initially

    # Filter dataset to 5 classes with 100 samples each
    dataset, selected_classes = filter_dataset(
        full_dataset, num_classes=5, samples_per_class=100
    )

    print(f"Training on classes: {selected_classes}")
    print(f"Total samples: {len(dataset)}")

    # Save the first image of each selected class
    save_class_images(dataset, selected_classes, "class_images")

    # VAE model path
    vae_model_path = "stabilityai/sd-vae-ft-mse"

    # Create preprocessed dataset
    vae = StableDiffusionVAE(vae_model_path)
    preprocessed_dataset = PreprocessedDataset(dataset, vae)

    # Create DataLoader
    train_dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        multiprocessing_context="spawn",
    )

    # DiT Configuration
    dit_config = DiTConfig(
        image_size=32,  # Latent space size (256 / 8)
        patch_size=2,  # Adjusted for the latent space size
        in_channels=4,  # Latent channels from VAE
        out_channels=4,  # Latent channels for VAE
        n_layer=8,  # Reduced number of layers for faster training
        n_head=8,  # Reduced number of heads
        n_embd=512,  # Reduced embedding dimension
        num_classes=5,  # Now we only have 5 classes
    )

    # Initialize the model
    model = DiTLightning(dit_config, vae_model_path)

    # Set up the trainer
    trainer = pl.Trainer(
        max_epochs=200,  # Increased epochs for overfitting
        logger=wandb_logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
    )

    # Start training
    trainer.fit(model, train_dataloader)
    wandb.finish()
