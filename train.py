import gc
import logging
import math
import multiprocessing
import os
import warnings

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from diffusers import DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from dataset import PreprocessedDataset, PreprocessedCatDataset
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
        self.max_lr_steps = 2000
        self.max_lr = 8e-4
        self.min_lr = 6e-5
        self.warmup_steps = 200

    def forward(self, x, t, class_labels):
        return self.net(x, t, class_labels)

    def get_lr(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        if it > self.max_lr_steps:
            return self.min_lr
        decay_ratio = (it - self.warmup_steps) / (self.max_lr_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure: bool = None,
    ):
        # Set the learning rate
        it = self.trainer.global_step
        lr = self.get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step(closure=optimizer_closure)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step >= self.max_lr_steps:
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group["lr"] = self.min_lr

    def setup(self, stage=None):
        # Ensure VAE is on the correct device and in eval mode
        self.vae.vae = self.vae.vae.to(self.device)
        self.vae.vae.eval()
        for param in self.vae.vae.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        images, class_labels, _ = batch

        # Ensure images are in bfloat16
        images = images.to(torch.bfloat16)

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

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("learning_rate", current_lr, on_step=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=0.1,
            betas=[0.9, 0.95],
            eps=1e-8,
        )
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
    wandb_logger = WandbLogger(project="dit-cats", log_model="all")

    vae = StableDiffusionVAE()

    logging.info("Loading the dataset...")
    original_dataset = load_dataset("huggan/cats", split="train")

    logging.info("Creating PreprocessedDataset...")
    preprocessed_dataset = PreprocessedCatDataset(
        dataset=original_dataset,
        size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

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
    model = DiTLightning(dit_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_cat",
        filename="dit-{epoch:02d}-{step}-{train_loss:.2f}",
        monitor="train_loss",
        every_n_train_steps=20,
        save_on_train_epoch_end=False,
    )

    # Set up the trainer
    trainer = pl.Trainer(
        max_epochs=150,
        logger=wandb_logger,
        precision="bf16-mixed",
        log_every_n_steps=1,
        accelerator="cuda",
        devices="8",
        strategy="ddp",
        callbacks=[checkpoint_callback],
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
    )

    # Start training
    trainer.fit(model, train_dataloader)
    wandb.finish()
