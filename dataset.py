import torch
import cv2 as cv
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms


from tqdm import tqdm
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')


def load_img(name):

    root = './images'
    name = osp.join(root, name)

    f = open(name)
    lines = f.readlines()
    imgs, lab = [], []

    random.shuffle(lines)

    for i in tqdm(range(len(lines)), desc = f"load {name}"):

        fn, label = lines[i].split(' ')

        im1 = cv.imread(osp.join(root, fn))
        im1 = cv.resize(im1, (256,256))
        #im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

        imgs.append(im1) 
        lab.append(int(label))  

    lab = np.asarray(lab, np.int32)
    imgs = np.stack(imgs)

    return imgs, lab 


class ImageMini(Dataset):
    def __init__(self, split, use_channels="RGB", pading=False):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------

        assert use_channels in ['RGB', 'RG', 'RB', 'BG', 'R', 'G', 'B'], f'Wrong Channel Name:{use_channels}'

        self.use_channels = use_channels
        self.padding = pading

        imgs, labs = load_img(f'{split}.txt')

        # original image
        self.imgs = torch.Tensor(imgs).permute(0,3,1,2).contiguous()        
        self.labs = torch.tensor(labs, dtype=torch.long)

        self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

        assert len(self.imgs) == len(self.labs), 'mismatched length!'

        self.nclass = int(torch.max(self.labs)) + 1
        print(f'Total data in {split} split: {len(self.imgs)}')
        print(f'Total Class in {split} split: {self.nclass}')
    

    def _padding(self,img):
        _, H, W = img.size()
        
        padding_map = {'RG':[0], 'RB':[2], 'BG':[1], 'R':[0,2], 'G':[0,1], 'B':[1,2]}

        use_channel = padding_map.get(self.use_channels)
        pad = torch.zeros(len(use_channel), H, W)

        img[use_channel,:,:] = pad

        return img
    

    def _select_channel(self,img):
        
        channel_map = {'RG':[1,2], 'RB':[0,1], 'BG':[0,2], 'R':[1], 'G':[2], 'B':[0]}

        use_channel = channel_map.get(self.use_channels)

        return img[use_channel,:,:]

            
    def __getitem__(self, index):

        img = self.transform(self.imgs[index])
        
        if len(self.use_channels) != 3:

            # use for naive testing
            if self.padding:
                img = self._padding(img)

            # use for dy_cnn training
            else:
                img = self._select_channel(img)
 
        lbl = self.labs[index]

        return img, lbl

    def __len__(self):

        return len(self.imgs)