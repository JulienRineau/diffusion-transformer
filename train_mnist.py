import gc
import logging
import math
import os
import warnings

import pytorch_lightning as pl
import torch
from diffusers import DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from dit import DiT, DiTConfig

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class DiTLightningMNIST(pl.LightningModule):
    def __init__(self, dit_config):
        super().__init__()
        self.save_hyperparameters()
        self.net = DiT(dit_config)
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
        it = self.trainer.global_step
        lr = self.get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step(closure=optimizer_closure)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step >= self.max_lr_steps:
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group["lr"] = self.min_lr

    def training_step(self, batch, batch_idx):
        images, class_labels = batch

        # Add channel dimension if necessary
        if images.shape[1] == 28:
            images = images.unsqueeze(1)

        # Normalize images to [-1, 1]
        images = images * 2 - 1

        # Sample noise to add to the images
        noise = torch.randn_like(images)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Add noise to the images according to the noise magnitude at each timestep
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.net(noisy_images, timesteps, class_labels)

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

    wandb.init(project="dit-mnist", save_code=True, mode="online")
    wandb_logger = WandbLogger(log_model=False)

    # Load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # DiT Configuration for MNIST
    dit_config = DiTConfig(
        image_size=28,  # MNIST image size
        patch_size=2,
        in_channels=1,  # MNIST is grayscale
        out_channels=1,
        n_layer=12,
        n_head=8,
        n_embd=384,
        num_classes=10,  # MNIST has 10 classes
    )

    # Initialize the model
    model = DiTLightningMNIST(dit_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_mnist_2",
        filename="dit-mnist-{epoch:02d}-{step}-{train_loss:.2f}",
        monitor="train_loss",
        every_n_train_steps=100,
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    # Set up the trainer
    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
        precision="bf16-mixed",
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        mnist_train,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # Start training
    trainer.fit(model, train_dataloader)
    wandb.finish()
