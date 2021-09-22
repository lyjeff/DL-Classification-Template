import os
from glob import glob
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

# image transform for train and test
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.Grayscale(3),
    transforms.ToTensor(),
    # transforms.GaussianBlur((5, 5), 1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# for training
class CatDataset(Dataset):
    def __init__(self, path):

        self.data = []

        for img_name in glob(os.path.join(path, '*.jpg')):
            label = 1.0 if os.path.basename(img_name).split('.')[0] == 'dog' else 0.0
            self.data.append((Image.open(img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = data_transforms(self.data[idx][0])
        return data, self.data[idx][1]

# for evaluating
class DogDataset(Dataset):
    def __init__(self, path, csv):

        self.data = []

        for img_id in csv['id']:
            img_path= os.path.join(path, f'{img_id}.jpg')
            self.data.append(Image.open(img_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = data_transforms(self.data[idx])
        return data