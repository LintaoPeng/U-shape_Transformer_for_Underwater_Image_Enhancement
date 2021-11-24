from torch.utils import data
import torchvision.transforms as transforms
import torch
import os
from PIL import Image

"""
Five K Dataset: 
    constructor params: 
        -list_file: path of a file with the list of images
        -raw_dir: path of the directory which contains raw images
        -expert_dir: path of the directory which contains expert images
        -training: if True, horizontal flip is applied
        -size: if is not None, resize is applied
        -filenames, if is true, __getitem__ returns also the name of the image taken  
"""

class FiveKDataset(data.Dataset):
    def __init__(self, list_file, raw_dir, expert_dir, training, size=None, filenames=False):
        join = os.path.join
        self.file_list = []
        with open(list_file) as f:
            for line in f:
                name = line.strip()
                if name:
                    p = (join(raw_dir, name), join(expert_dir, name), name)
                    self.file_list.append(p)
        self.filenames = filenames
        transformation=[]
        if size is not None:
          transformation.append(transforms.Resize((size,size)))
        if training:
          transformation.append(transforms.RandomHorizontalFlip(0.5))
        transformation.append(transforms.ToTensor())
        self.transform=transforms.Compose(transformation)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        raw = Image.open(self.file_list[index][0])
        expert =  Image.open(self.file_list[index][1])
        raw_exp = self.transform(torch.stack([raw, expert]))
        if self.filenames:
            return raw_exp[0],raw_exp[1], self.file_list[index][2]
        else:
            return raw_exp[0],raw_exp[1]


def _test():
    train_list = "/content/drive/MyDrive/fivek/train1+2-list.txt"
    raw_dir = "/content/drive/MyDrive/fivek/raw"
    expert_dir = "/content/drive/MyDrive/fivek/expC"
    dataset = FiveKDataset(train_list, raw_dir, expert_dir, True, None)
    loader = torch.utils.data.DataLoader(dataset, 10, shuffle=True,num_workers=16)
    for raw, expert in loader:
        print(raw.size(), expert.size())



def _test():
    train_list = "../data/train1+2-list.txt"
    raw_dir = "/mnt/data/dataset/fivek/siggraph2018/256x256/raw"
    expert_dir = "/mnt/data/dataset/fivek/siggraph2018/256x256/expC"
    dataset = FiveKDataset(train_list, raw_dir, expert_dir, True, None)
    loader = torch.utils.data.DataLoader(dataset, 10, shuffle=True)
    for raw, expert in loader:
        print(raw.size(), expert.size())


if __name__ == '__main__':
    _test()
