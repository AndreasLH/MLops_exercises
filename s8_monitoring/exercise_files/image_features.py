from PIL import Image
import requests
# requires transformers package: pip install transformers
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import DataLoader
import pandas as pd

# dataset_train = CIFAR10(root='/data', train=False, download=True)
dataset_train2 = SVHN(root='/data', split='train', download=True)

train_loader = DataLoader(dataset_train2, batch_size=2, 
                              shuffle=True, num_workers=0)
        
class Svhn_dataloader(DataLoader):
    def __init__(self):
        super().__init__()
        self.dataset = SVHN(root='/data', split='train', download=True)
    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]
    def collate_fn(self, batch):
        return []

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# set either text=None or images=None when only the other is needed
# for images, labels in train_loader:
img = next(iter(train_loader))
inputs = processor(images=img, return_tensors="pt", padding=True)

img_features = model.get_image_features(inputs['pixel_values'])
# text_features = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
pd.DataFrame(img_features.detach().numpy()).to_csv('s8_monitoring/exercise_files/image_features.csv', index=False)