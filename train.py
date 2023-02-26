import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNET

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
)

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMG_WIDTH = 224
IMG_HEIGHT = 224
PIN_MEMORY = True
LOAD_MODEL = True
IMG_DIR = os.path.join(os.curdir, "dataset", "frames")
MASK_DIR = os.path.join(os.curdir, "dataset", "masks")

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, ncols=100)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast_mode.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step()

        loop.set_postfix(loss=loss.item())



def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        IMG_DIR, MASK_DIR, BATCH_SIZE, train_transforms, val_transforms, NUM_WORKERS, PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("unet_checkpoint.pth.tar"))
        
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss, scaler)

        checkpoints = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoints)
        check_accuracy(val_loader, model, device=DEVICE)


if __name__ == '__main__':
    main()
