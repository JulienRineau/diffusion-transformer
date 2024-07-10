from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import random


def load_imagenet_dataset(num_samples=100, split="validation"):
    """
    Load a subset of the ImageNet dataset from Hugging Face.

    Args:
    num_samples (int): Number of samples to load. Default is 100.
    split (str): Dataset split to use. Default is "validation".

    Returns:
    list: A list of tuples, each containing (image, label, class_name).
    """
    # Load the ImageNet dataset
    dataset = load_dataset(
        "imagenet-1k", split=split, trust_remote_code=True, streaming=True
    )

    # Load the class labels
    labels = dataset.features["label"].names

    # Define a transform for the images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Randomly sample 'num_samples' from the dataset
    # sampled_data = random.sample(range(len(dataset)), num_samples)
    sampled_data = random.sample(1000, num_samples)

    annotated_images = []
    for idx in sampled_data:
        # Get the image and its label
        sample = dataset[idx]
        image = sample["image"]
        label = sample["label"]

        # Get the class name
        class_name = labels[label]

        # Apply the transform to the image
        image_tensor = transform(image)

        annotated_images.append((image_tensor, label, class_name))

    return annotated_images


if __name__ == "__main__":
    dataset = load_imagenet_dataset(num_samples=10)
    for image, label, class_name in dataset:
        print(f"Image shape: {image.shape}, Label: {label}, Class: {class_name}")
