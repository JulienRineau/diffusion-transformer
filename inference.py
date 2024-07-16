import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from dit import DiT, DiTConfig
from train import DiTLightning
from vae import StableDiffusionVAE


def sample_dit(
    model,
    noise_scheduler,
    n_inference_steps=50,
    batch_size=8,
    image_size=(4, 32, 32),  # Updated for latent space size
    device="cuda",
    class_labels=None,
):
    """
    Sample images using the DiT model with class conditioning for a single-class dataset.

    Args:
    model (nn.Module): The trained DiT model
    noise_scheduler (DDPMScheduler): The noise scheduler
    n_inference_steps (int): Number of inference steps
    batch_size (int): Number of images to generate
    image_size (tuple): Size of the image in latent space (channels, height, width)
    device (str): Device to run the model on ('cuda' or 'cpu')
    class_labels (torch.Tensor): Tensor of class labels for conditioning

    Returns:
    tuple: (step_history, pred_output_history)
    """
    noise_scheduler.set_timesteps(n_inference_steps)

    # Start from random noise
    x = torch.randn(batch_size, *image_size).to(device)
    step_history = [x.detach().cpu()]
    pred_output_history = []

    if class_labels is None:
        class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        class_labels = class_labels.to(device)

    model.eval()
    with torch.no_grad():
        for t in noise_scheduler.timesteps:
            # Expand the latents if we are doing classifier free guidance
            model_input = torch.cat([x] * 2)
            model_input = noise_scheduler.scale_model_input(model_input, t)

            timesteps = torch.full(
                (batch_size * 2,), t, device=device, dtype=torch.long
            )
            class_labels_input = class_labels.repeat(2)

            # Predict the noise residual
            noise_pred = model(model_input, timesteps, class_labels_input)
            noise_pred = noise_pred[:batch_size]  # Take only the unconditional output

            # Compute the previous noisy sample x_t -> x_t-1
            x = noise_scheduler.step(noise_pred, t, x).prev_sample

            step_history.append(x.detach().cpu())
            pred_output_history.append(noise_pred.detach().cpu())

    return step_history, pred_output_history


def visualize_sampling_process(
    vae,
    step_history,
    pred_output_history,
    class_labels,
    save_path="dit_cat_sampling_visualization.png",
    steps_to_visualize=None,
):
    """
    Visualize the sampling process for the cat dataset in a single image.

    Args:
    vae (StableDiffusionVAE): The VAE model for decoding latents
    step_history (list): List of tensors representing the sampling steps
    pred_output_history (list): List of tensors representing the model predictions
    class_labels (torch.Tensor): Tensor of class labels used for conditioning
    save_path (str): Path to save the visualization
    steps_to_visualize (list): List of specific steps to visualize
    """
    n_steps = len(pred_output_history)
    n_samples = step_history[0].shape[0]

    if steps_to_visualize is None:
        steps_to_visualize = list(range(n_steps))

    n_visualized_steps = len(steps_to_visualize)

    fig, axs = plt.subplots(n_visualized_steps, 2, figsize=(12, 2 * n_visualized_steps))
    plt.subplots_adjust(hspace=0.3)

    if n_visualized_steps == 1:
        axs = [axs]

    axs[0][0].set_title("Generated Image")
    axs[0][1].set_title("Predicted Noise")

    for idx, i in enumerate(steps_to_visualize):
        # Decode latents to images
        with torch.no_grad():
            images = vae.decode(step_history[i].to(vae.vae.device))

        # Plot generated image
        images = (images + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
        axs[idx][0].imshow(make_grid(images.cpu(), nrow=4).permute(1, 2, 0))
        axs[idx][0].axis("off")

        # Plot predicted noise
        axs[idx][1].imshow(
            make_grid(pred_output_history[i], nrow=4).permute(1, 2, 0).mean(dim=2),
            cmap="viridis",
        )
        axs[idx][1].axis("off")

        axs[idx][0].set_ylabel(f"Step {n_steps - i}")

    # Add class labels to the plot
    fig.suptitle(f"Generated Cat Images", fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from checkpoint
    checkpoint_path = "checkpoints_cat/dit-epoch=27-step=2180-train_loss=0.26.ckpt"  # Update this path
    model = DiTLightning.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    print("Model loaded successfully")

    # Get the noise scheduler from the model
    noise_scheduler = model.noise_scheduler

    # Set inference parameters
    n_inference_steps = 1000
    batch_size = 4
    image_size = (4, 32, 32)  # Latent space size for 256x256 images

    # Set class labels for conditioning (all zeros for single-class dataset)
    class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Perform sampling
    step_history, pred_output_history = sample_dit(
        model.net,
        noise_scheduler,
        n_inference_steps,
        batch_size,
        image_size,
        device,
        class_labels,
    )
    print("Sampling completed")

    # Specify the steps you want to visualize
    steps_to_visualize = [0, 200, 400, 600, 800, 999]

    vae = StableDiffusionVAE().to(device)

    # Then update the visualization call:
    visualize_sampling_process(
        vae,
        step_history,
        pred_output_history,
        class_labels,
        "dit_cat_sampling_visualization.png",
        steps_to_visualize=steps_to_visualize,
    )

    print("Process completed")
