import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class BUDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', test_size=0.3, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        self.classes = [cls for cls in os.listdir(root_dir) 
                        if not cls.startswith('.') and os.path.isdir(os.path.join(root_dir, cls))]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        random.seed(seed)
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            img_names = [f for f in os.listdir(cls_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
            random.shuffle(img_names)
            
            split_idx = int(len(img_names) * (1 - test_size))
            selected_names = img_names[:split_idx] if mode == 'train' else img_names[split_idx:]
            
            for img_name in selected_names:
                self.images.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls])
                
        print(f"{mode}集: 总样本数={len(self.images)}, 类别分布={[self.labels.count(i) for i in range(len(self.classes))]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label 

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

def get_data_loaders(root_dir, batch_size=4):
    train_transform, test_transform = get_transforms()
    train_dataset = BUDataset(root_dir, transform=train_transform, mode='train')
    test_dataset = BUDataset(root_dir, transform=test_transform, mode='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.classes