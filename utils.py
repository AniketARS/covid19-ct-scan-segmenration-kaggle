import os
import random

import torch
from covid_dataset import CovidDataset
from torch.utils.data import DataLoader, random_split

def save_checkpoint(state, filename="unet_checkpoint.pth.tar"):
    print("Saving Checkpoints...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading Checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(
        img_dir, mask_dir, batch_size, train_transforms, val_transforms, num_workers, pin_memory
):
    files = os.listdir(img_dir)
    split = int(len(files) * 0.9)
    random.shuffle(files)
    train_images, val_images = files[:split], files[split:]
    train_ds = CovidDataset(
        image_dir=img_dir, mask_dir=mask_dir, images=train_images, transform=train_transforms
    )
    val_ds = CovidDataset(
        image_dir=img_dir, mask_dir=mask_dir, images=val_images, transform=val_transforms
    )

    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=False
    )
    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds==y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds*y).sum()) / ((preds+y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice Score: {dice_score/len(loader)}")
    model.train()

