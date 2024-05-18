import os
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from custom_transforms import *

class CityScapes(Dataset):
    def __init__(self, data_path, cropsize= (640, 480), randomscale= (0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), mode= 'train', *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ['train', 'val', 'test', 'trainval']
        self.mode = mode
        print(f'loading {self.mode} dataset')
        self.label_to_ignore = 255

        with open(os.path.join(data_path, 'cityscapes_info.json'), 'r') as info_file:
            labels_info = json.load(info_file)
        self.label_mapping = {el['id']: el['trainId'] for el in labels_info}


        ## loading images

        self.images = {}
        image_names = []

        print(f'data_path: {data_path}')

        li8b_path = os.path.join(data_path, 'data', 'leftImg8bit', mode)
        print(li8b_path)

        folders = os.listdir(li8b_path)
        print(folders)

        for folder_name in folders:
            folder_path = os.path.join(li8b_path, folder_name)
            file_names = os.listdir(folder_path)
            current_image_names = [elem.replace('_leftImg8bit.png', '') for elem in file_names]
            file_paths = [os.path.join(folder_path, elem) for elem in file_names]
            image_names.extend(current_image_names)
            self.images.update(dict(zip(current_image_names, file_paths)))

        print(next(iter(self.images.items())))


        ## loading labels

        self.labels = {}
        label_names = []

        gt_path = os.path.join(data_path, 'data', 'gtFine', mode)
        print(gt_path)

        folders = os.listdir(gt_path)
        print(folders)

        for folder_name in folders:
            folder_path = os.path.join(gt_path, folder_name)
            file_names = [elem for elem in os.listdir(folder_path) if 'labelIds' in elem]
            current_label_names = [elem.replace('_gtFine_labelIds.png', '') for elem in file_names]
            file_paths = [os.path.join(folder_path, elem) for elem in file_names]
            label_names.extend(current_label_names)
            self.labels.update(dict(zip(current_label_names, file_paths)))

        print(next(iter(self.labels.items())))

        self.image_names = image_names
        self.len = len(self.image_names)
        print(f'{self.len} images loaded from {self.mode}')

        assert set(image_names) == set(label_names)
        assert set(self.image_names) == set(self.images.keys())
        assert set(self.image_names) == set(self.labels.keys())


        ## preprocessing

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        
        self.transformations_on_train = custom_Compose([
            custom_ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5),
            custom_HorizontalFlip(),
            custom_RandomScale(randomscale),
            custom_RandomCrop(cropsize),
        ])


    def convert_labels(self, label):
        for k, v in self.label_mapping.items():
            label[label == k] = v
        return label


    def __getitem__(self, idx):
        idx_key = self.image_names[idx]
        image_path = self.images[idx_key]
        label_path = self.labels[idx_key]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        if self.mode in ['train', 'trainval']:
            image_label = {'image': image, 'label': label}
            image_label = self.transformations_on_train(image_label)
            image, label = image_label['image'], image_label['label']
        image = self.to_tensor(image)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return image, label


    def __len__(self):
        return self.len