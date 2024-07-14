import gc
import multiprocessing
import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

import torch
from datasets import load_dataset
from diffusers import DDPMScheduler
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from dataset import PreprocessedDataset
from dit import DiT, DiTConfig
from vae import StableDiffusionVAE

multiprocessing.set_start_method("spawn", force=True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class DiTLightning(pl.LightningModule):
    def __init__(self, dit_config):
        super().__init__()
        self.save_hyperparameters()
        self.net = DiT(dit_config)
        self.vae = StableDiffusionVAE()
        self.num_train_timesteps = 1000
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )

    def forward(self, x, t, class_labels):
        return self.net(x, t, class_labels)

    def training_step(self, batch, batch_idx):
        images, class_labels, _ = batch

        # Move VAE to the same device as the input
        self.vae = self.vae.to(images.device)

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
        if not torch.isfinite(loss):
            print(f"Non-finite loss detected: {loss}")
            return None

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


def filter_dataset(dataset, num_classes=5, samples_per_class=None):
    """
    Filter the dataset to include a specified number of classes and samples per class.

    Args:
    dataset: The original dataset
    num_classes (int): Number of classes to include
    samples_per_class (int or None): Number of samples per class. If None, include all samples.

    Returns:
    Subset: A subset of the original dataset
    list: List of selected class labels
    """
    # Get unique classes
    all_classes = set(label for _, label, _ in dataset)
    selected_classes = list(all_classes)[:num_classes]

    # Filter dataset
    filtered_indices = [
        i for i, (_, label, _) in enumerate(dataset) if label in selected_classes
    ]

    if samples_per_class is None:
        final_indices = filtered_indices
    else:
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


def clean_cuda_memory():
    print("Cleaning CUDA...")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    clean_cuda_memory()
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_logger = WandbLogger(project="dit", log_model="all")

    # Create preprocessed dataset
    vae = StableDiffusionVAE()

    preprocessed_dataset = PreprocessedDataset(
        "imagenet_subset_shards", size=256, device="cpu"
    )

    # DiT Configuration
    dit_config = DiTConfig(
        image_size=32,  # Latent space size (256 / 8)
        patch_size=2,  # Adjusted for the latent space size
        in_channels=4,  # Latent channels from VAE
        out_channels=4,  # Latent channels for VAE
        n_layer=12,  # Reduced number of layers for faster training
        n_head=12,  # Reduced number of heads
        n_embd=768,  # Embedding dimension
        num_classes=preprocessed_dataset.num_classes,  # Now we only have 5 classes
    )

    # Initialize the model
    model = DiTLightning(dit_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="dit-{epoch:02d}-{step}-{train_loss:.2f}",
        monitor="train_loss",
        save_top_k=50,
        every_n_train_steps=100,
        save_on_train_epoch_end=False,
    )

    # Set up the trainer
    trainer = pl.Trainer(
        max_epochs=5,  # Increased epochs for overfitting
        logger=wandb_logger,
        precision=16,
        log_every_n_steps=10,
        accelerator="gpu",
        devices="8",
        accumulate_grad_batches=8,
        strategy=DDPStrategy(),
        callbacks=[checkpoint_callback],
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
    )

    # Start training
    trainer.fit(model, train_dataloader)
    wandb.finish()
