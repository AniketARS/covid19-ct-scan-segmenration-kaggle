import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CovidDataset(Dataset):

    def __init__(self, image_dir, mask_dir, images, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = images

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        image = image.transpose(2, 0, 1)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augementation = self.transform(image=image, mask=mask)
            image = augementation['image']
            mask = augementation['mask']

        return image, mask

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    dataset = CovidDataset(image_dir="./dataset/frames", mask_dir="./dataset/masks")
    image, mask = dataset[5]
    print(image.shape, mask.shape)
