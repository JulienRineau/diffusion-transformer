import gc
import logging
import multiprocessing
import os
import warnings
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from diffusers import DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from dataset import PreprocessedCatDataset, PreprocessedDataset
from dit import DiT, DiTConfig
from vae import StableDiffusionVAE

multiprocessing.set_start_method("spawn", force=True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@dataclass
class TrainerConfig:
    batch_size: int = 32
    lr: int = 1e-4


class DiTLightning(pl.LightningModule):
    def __init__(
        self,
        dit_config,
        trainer_config: TrainerConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = DiT(dit_config)
        self.vae = StableDiffusionVAE()
        self.num_train_timesteps = 1000
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )
        self.lr = trainer_config.lr
        self.batch_size = trainer_config.batch_size

    def forward(self, x, t, class_labels):
        return self.net(x, t, class_labels)

    def setup(self, stage=None):
        # Ensure VAE is on the correct device and in eval mode
        self.vae.vae = self.vae.vae.to(self.device)
        self.vae.vae.eval()
        for param in self.vae.vae.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        images, class_labels, _ = batch

        images = images.to(torch.bfloat16)
        latents = self.vae.encode(images)
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0, self.num_train_timesteps, (latents.shape[0],), device=self.device
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.net(noisy_latents, timesteps, class_labels)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        if not torch.isfinite(loss):
            print(f"Non-finite loss detected: {loss}")
            return None

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        return optimizer


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

    # Configure Wandb to log metrics but not save artifacts
    wandb.init(project="dit-cats", save_code=True, mode="online")
    wandb_logger = WandbLogger(log_model=False)

    vae = StableDiffusionVAE()

    logging.info("Loading the dataset...")
    original_dataset = load_dataset("huggan/cats", split="train")

    logging.info("Creating PreprocessedDataset...")
    preprocessed_dataset = PreprocessedCatDataset(
        dataset=original_dataset,
        size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer_config = TrainerConfig(batch_size=32, lr=1e-4)

    # DiT Configuration
    dit_config = DiTConfig(
        image_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        n_layer=12,
        n_head=12,
        n_embd=768,
        num_classes=preprocessed_dataset.num_classes,
    )

    # Initialize the model
    model = DiTLightning(dit_config, trainer_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_cat_reddit_fix",
        filename="dit-{epoch:02d}-{step}-{train_loss:.2f}",
        monitor="train_loss",
        every_n_train_steps=40,
        save_top_k=4,
        save_on_train_epoch_end=False,
    )

    # Set up the trainer
    trainer = pl.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        precision="bf16-mixed",
        log_every_n_steps=1,
        accelerator="cuda",
        devices="8",
        strategy="ddp",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
    )

    # Start training
    trainer.fit(model, train_dataloader)
    wandb.finish()
