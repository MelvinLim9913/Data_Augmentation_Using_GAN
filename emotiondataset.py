from PIL import Image
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.image = img_list
        self.label = label_list
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.image[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label[idx]

        return img, label

    def __len__(self):
        return len(self.image)
