import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import random
# from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

data_dir =sys.argv[1] 
model_ckpt_dir = sys.argv[2] 

# Channels in the model after each layer
input_channel = 3
channels_after_conv1 = 32
channels_after_layer1 = 32
channels_after_layer2 = 64
channels_after_layer3 = 128

# Hyperparameters
learning_rate = 0.1
weight_decay = 5e-4
n = 2
batch_size = 64
num_epochs = 40

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, n=2, num_classes=100):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, channels_after_conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_after_conv1)
        self.relu = nn.ReLU(inplace=True)
        
        # First stage: n layers with 32 filters, feature map 224x224
        self.layer1 = self._make_layer(channels_after_conv1, channels_after_layer1, n, stride=1)
        
        # Second stage: n layers with 64 filters, feature map 112x112
        self.layer2 = self._make_layer(channels_after_layer1, channels_after_layer2, n, stride=2)
        
        # Third stage: n layers with 128 filters, feature map 56x56
        self.layer3 = self._make_layer(channels_after_layer2, channels_after_layer3, n, stride=2)
        
        # Global average pooling and FC layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels_after_layer3, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Transforms on train
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform

def load_datasets(data_dir):
    train_transform = get_transforms()
    
    train_dir = data_dir
    train_dataset = ImageFolder(train_dir, transform=train_transform)

    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # print(f"Found {len(class_to_idx)} classes")
    return train_dataset, idx_to_class

# def validate_model(model, val_loader, criterion, device='cuda'):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     val_loss = running_loss / len(val_loader.dataset)
#     val_acc = 100.0 * correct / total
#     return val_loss, val_acc

# def train_model(model, train_loader, num_epochs=50, device='cuda', val_loader=None):
def train_model(model, train_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, 
                         weight_decay=weight_decay, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        # print(f'Epoch {epoch+1}/{num_epochs} | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% | LR: {current_lr:.2e}')
        
        ckpt_path = os.path.join(model_ckpt_dir, 'resnet_model.pth')
        torch.save(model.state_dict(), ckpt_path)
        # print(f'Epoch {epoch+1}/{num_epochs}')
        # # ----------------- Validation Step ----------------- #
        # if val_loader is not None:
        #     val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        #     print(f'Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    return model

# Main function (only training with command-line arguments)
def main():
    # if len(sys.argv) != 3:
    #     print("Usage: python train.py <train_data_dir> <model_ckpt_dir>")
    #     sys.exit(1)
        
    # data_dir = "/kaggle/input/dataset2/Butterfly/Butterfly/train"  # sys.argv[1]
    # model_ckpt_dir = "/kaggle/working"  # sys.argv[2]

    if not os.path.exists(model_ckpt_dir):
        os.makedirs(model_ckpt_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')

    train_dataset, idx_to_class = load_datasets(data_dir)
    
    # # Attempt to load validation dataset
    # # Assumes that the validation folder is located by replacing 'train' with 'val' in the provided data_dir.
    # val_dir = data_dir.replace('train', 'valid')
    # if os.path.exists(val_dir):
    #     val_dataset = ImageFolder(val_dir, transform=get_transforms())
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    #     print(f'Validation set: {len(val_dataset)} images')
    # else:
    #     val_loader = None
    #     print("Validation directory not found. Skipping validation.")
    
    # Update num_classes based on actual dataset
    num_classes = len(train_dataset.classes)
    
    # Create model
    model = ResNet(n=n, num_classes=num_classes)
    # print(f'Created ResNet model with {6*n+2} layers')
    
    # Create training data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # print(f'Training set: {len(train_dataset)} images')
    
    # Train model (with validation if available)v
    trained_model = train_model(model, train_loader, num_epochs=num_epochs, device=device)
    
    # Save final model to the checkpoint directory
    ckpt_path = os.path.join(model_ckpt_dir, 'resnet_model.pth')
    torch.save(trained_model.state_dict(), ckpt_path)
    # print(f'Training complete, model saved at {ckpt_path}')

if __name__ == '__main__':
    main()
