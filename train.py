import gc
import logging
import multiprocessing
import os
import warnings
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import wandb
from datasets import load_dataset
from diffusers import DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import PreprocessedDataset
from dit import DiT, DiTConfig

multiprocessing.set_start_method("spawn", force=True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@dataclass
class TrainerConfig:
    batch_size: int = 32
    lr: float = 1e-4


class DiTLightning(pl.LightningModule):
    def __init__(self, dit_config, trainer_config: TrainerConfig):
        super().__init__()
        self.save_hyperparameters()
        self.net = DiT(dit_config)
        self.num_train_timesteps = 1000
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )
        self.lr = trainer_config.lr
        self.batch_size = trainer_config.batch_size

    def forward(self, x, t, class_labels):
        return self.net(x, t, class_labels)

    def training_step(self, batch, batch_idx):
        img_tensor, latent_tensor, label, _ = batch

        latents = latent_tensor.to(torch.bfloat16)
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0, self.num_train_timesteps, (latents.shape[0],), device=self.device
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.net(noisy_latents, timesteps, label)

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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)


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

    wandb.init(project="dit-cats", save_code=True, mode="online")
    wandb_logger = WandbLogger(log_model=True)

    logging.info("Loading the dataset...")
    original_dataset = load_dataset("huggan/cats", split="train")

    logging.info("Creating PreprocessedDataset...")
    preprocessed_dataset = PreprocessedDataset(
        dataset=original_dataset,
        size=256,
        device=device,
        shard_dir="cat_dataset_shards",
        num_shards=10,
    )

    trainer_config = TrainerConfig(batch_size=64, lr=1e-4)

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

    model = DiTLightning(dit_config, trainer_config)

    checkpoint_dir = "checkpoints_cat"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="dit-{epoch:02d}-{step}-{train_loss:.2f}",
        monitor="train_loss",
        mode="min",
        save_top_k=10,
        save_on_train_epoch_end=True,
        save_last=True,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=400,
        logger=wandb_logger,
        precision="bf16-mixed",
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=4,
    )

    train_dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
    )

    # Check for the latest checkpoint
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
            if not os.path.exists(latest_checkpoint):
                latest_checkpoint = os.path.join(
                    checkpoint_dir, sorted(checkpoints)[-1]
                )

    if latest_checkpoint and os.path.isfile(latest_checkpoint):
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.fit(model, train_dataloader, ckpt_path=latest_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.fit(model, train_dataloader)

    wandb.finish()
