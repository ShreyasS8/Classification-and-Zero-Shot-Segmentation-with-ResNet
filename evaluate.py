import os
import sys
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
# from tqdm import tqdm

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

INPUT_CHANNEL = 3
CHANNELS_AFTER_CONV1 = 32
CHANNELS_AFTER_LAYER1 = 32
CHANNELS_AFTER_LAYER2 = 64
CHANNELS_AFTER_LAYER3 = 128
NUM_CLASSES = 100

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, n=2, num_classes=NUM_CLASSES):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNEL, CHANNELS_AFTER_CONV1,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(CHANNELS_AFTER_CONV1)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(CHANNELS_AFTER_CONV1, CHANNELS_AFTER_LAYER1, n, stride=1)
        self.layer2 = self._make_layer(CHANNELS_AFTER_LAYER1, CHANNELS_AFTER_LAYER2, n, stride=2)
        self.layer3 = self._make_layer(CHANNELS_AFTER_LAYER2, CHANNELS_AFTER_LAYER3, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(CHANNELS_AFTER_LAYER3, num_classes)
        # Weight initialization
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
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
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

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

HIGH_QUANTILE = 0.950
LOW_QUANTILE  = 0.050
GRABCUT_ITER  = 5

def compute_saliency(model, input_tensor, device):
    model.eval()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_()
    output = model(input_tensor)
    top_class = output.argmax(dim=1)
    score = output[0, top_class]
    model.zero_grad()
    score.backward()
    saliency, _ = torch.max(input_tensor.grad.abs(), dim=1)
    return saliency.squeeze().cpu().detach().numpy()

def run_grabcut(img, saliency, high_quantile=HIGH_QUANTILE, low_quantile=LOW_QUANTILE, iterations=GRABCUT_ITER):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        if img.ndim == 4:
            img = np.squeeze(img, axis=0)
    img = np.ascontiguousarray(img.astype(np.uint8))
    
    high_thresh = np.quantile(saliency, high_quantile)
    low_thresh  = np.quantile(saliency, low_quantile)
    
    init_mask = np.full(saliency.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    init_mask[saliency >= high_thresh] = cv2.GC_FGD
    init_mask[saliency <= low_thresh]  = cv2.GC_BGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img, init_mask, None, bgdModel, fgdModel, iterations, mode=cv2.GC_INIT_WITH_MASK)
    
    final_mask = np.where((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    return final_mask

def main(model_ckpt_path, test_imgs_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    model = ResNet(n=2, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    # print(f"Model loaded from {model_ckpt_path}")

    idx_to_class = {
    0: 'ADONIS', 1: 'AFRICAN GIANT SWALLOWTAIL', 2: 'AMERICAN SNOOT', 3: 'AN 88', 4: 'APPOLLO', 5: 'ARCIGERA FLOWER MOTH', 6: 'ATALA', 7: 'ATLAS MOTH', 8: 'BANDED ORANGE HELICONIAN', 9: 'BANDED PEACOCK', 
    10: 'BANDED TIGER MOTH', 11: 'BECKERS WHITE', 12: 'BIRD CHERRY ERMINE MOTH', 13: 'BLACK HAIRSTREAK', 14: 'BLUE MORPHO', 15: 'BLUE SPOTTED CROW', 16: 'BROOKES BIRDWING', 17: 'BROWN ARGUS', 18: 'BROWN SIPROETA', 19: 'CABBAGE WHITE', 
    20: 'CAIRNS BIRDWING', 21: 'CHALK HILL BLUE', 22: 'CHECQUERED SKIPPER', 23: 'CHESTNUT', 24: 'CINNABAR MOTH', 25: 'CLEARWING MOTH', 26: 'CLEOPATRA', 27: 'CLODIUS PARNASSIAN', 28: 'CLOUDED SULPHUR', 29: 'COMET MOTH', 
    30: 'COMMON BANDED AWL', 31: 'COMMON WOOD-NYMPH', 32: 'COPPER TAIL', 33: 'CRECENT', 34: 'CRIMSON PATCH', 35: 'DANAID EGGFLY', 36: 'EASTERN COMA', 37: 'EASTERN DAPPLE WHITE', 38: 'EASTERN PINE ELFIN', 39: 'ELBOWED PIERROT', 
    40: 'EMPEROR GUM MOTH', 41: 'GARDEN TIGER MOTH', 42: 'GIANT LEOPARD MOTH', 43: 'GLITTERING SAPPHIRE', 44: 'GOLD BANDED', 45: 'GREAT EGGFLY', 46: 'GREAT JAY', 47: 'GREEN CELLED CATTLEHEART', 48: 'GREEN HAIRSTREAK', 49: 'GREY HAIRSTREAK', 
    50: 'HERCULES MOTH', 51: 'HUMMING BIRD HAWK MOTH', 52: 'INDRA SWALLOW', 53: 'IO MOTH', 54: 'Iphiclus sister', 55: 'JULIA', 56: 'LARGE MARBLE', 57: 'LUNA MOTH', 58: 'MADAGASCAN SUNSET MOTH', 59: 'MALACHITE', 
    60: 'MANGROVE SKIPPER', 61: 'MESTRA', 62: 'METALMARK', 63: 'MILBERTS TORTOISESHELL', 64: 'MONARCH', 65: 'MOURNING CLOAK', 66: 'OLEANDER HAWK MOTH', 67: 'ORANGE OAKLEAF', 68: 'ORANGE TIP', 69: 'ORCHARD SWALLOW', 
    70: 'PAINTED LADY', 71: 'PAPER KITE', 72: 'PEACOCK', 73: 'PINE WHITE', 74: 'PIPEVINE SWALLOW', 75: 'POLYPHEMUS MOTH', 76: 'POPINJAY', 77: 'PURPLE HAIRSTREAK', 78: 'PURPLISH COPPER', 79: 'QUESTION MARK', 
    80: 'RED ADMIRAL', 81: 'RED CRACKER', 82: 'RED POSTMAN', 83: 'RED SPOTTED PURPLE', 84: 'ROSY MAPLE MOTH', 85: 'SCARCE SWALLOW', 86: 'SILVER SPOT SKIPPER', 87: 'SIXSPOT BURNET MOTH', 88: 'SLEEPY ORANGE', 89: 'SOOTYWING', 
    90: 'SOUTHERN DOGFACE', 91: 'STRAITED QUEEN', 92: 'TROPICAL LEAFWING', 93: 'TWO BARRED FLASHER', 94: 'ULYSES', 95: 'VICEROY', 96: 'WHITE LINED SPHINX MOTH', 97: 'WOOD SATYR', 98: 'YELLOW SWALLOW TAIL', 99: 'ZEBRA LONG WING'
}

    transform = get_transform()
    seg_maps_folder = os.path.join(os.getcwd(), "seg_maps")
    os.makedirs(seg_maps_folder, exist_ok=True)
    
    submission_data = []  # List to hold rows for CSV: (image_name, label)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    test_images = [f for f in os.listdir(test_imgs_dir) if f.lower().endswith(valid_extensions)]
    
    for img_name in test_images:
        img_path = os.path.join(test_imgs_dir, img_name)
        
        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        
        image_tensor = transform(image_pil)
        image_tensor = image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor.to(device))
            _, predicted = torch.max(output, 1)
            pred_label = predicted.item()

        class_name = idx_to_class[pred_label]
        submission_data.append((img_name, class_name))

        saliency_map = compute_saliency(model, image_tensor, device)
        
        orig_image = cv2.imread(img_path)
        if orig_image is None:
            orig_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        grabcut_mask = run_grabcut(orig_image, saliency_map)
        kernel = np.ones((5, 5), np.uint8)
        refined_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_CLOSE, kernel)
        
        seg_map_path = os.path.join(seg_maps_folder, img_name)
        mask_to_save = (refined_mask * 255).astype(np.uint8)
        cv2.imwrite(seg_map_path, mask_to_save)
    
    csv_filename = "submission.csv"
    with open(csv_filename, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_name", "label"])
        for row in submission_data:
            csvwriter.writerow(row)
    
    # print(f"Processing complete. Submission CSV saved as '{csv_filename}'.")
    # print(f"Segmentation maps saved in folder '{seg_maps_folder}'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <model_ckpt path> <test-imgs dir>")
        sys.exit(1)
    model_ckpt_path = sys.argv[1]
    test_imgs_dir = sys.argv[2]
    main(model_ckpt_path, test_imgs_dir)
