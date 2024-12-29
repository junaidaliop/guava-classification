import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

class GuavaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images, organized in subfolders per class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        self.transform = transform

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for image_name in os.listdir(cls_folder):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(cls_folder, image_name))
                    self.labels.append(cls_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

def get_train_transforms():
    return A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_test_transforms():
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_data(config):
    """
    Args:
        config (dict): Configuration dictionary containing data paths and batch size.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """
    train_transform = get_train_transforms()
    val_transform = get_val_test_transforms()
    test_transform = get_val_test_transforms()

    train_dataset = GuavaDataset(root_dir=config['data']['train_dir'], transform=train_transform)
    val_dataset = GuavaDataset(root_dir=config['data']['val_dir'], transform=val_transform)
    test_dataset = GuavaDataset(root_dir=config['data']['test_dir'], transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
