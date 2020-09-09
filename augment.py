import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

trans = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.RandomAffine(degrees=(0, -0), translate=(0.35, 0.35), fillcolor=(255, 255, 255)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomApply([transforms.RandomResizedCrop((120, 120), scale=(0.5, 1.0), ratio=(0.75, 1.333), interpolation=2)], p=0.02),
    transforms.ToTensor()
])

root = r"C:\Users\sid_a\OneDrive\Documents\Pokemon Test\data_split_no_augment\val"
data = torchvision.datasets.ImageFolder(root=root, transform=trans)
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

n = 0
for _ in range(4):
    for img, label in iter(data_loader):
        save_image(img, r"C:\Users\sid_a\OneDrive\Documents\Pokemon Test\data_final_2\4\img" + str(n) + ".jpg")
        n += 1
