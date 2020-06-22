import json
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import h5py


img = None
img_info = {}


def gqa_obj_feature_loader():
    global img, img_info
    if img is not None:
        return img, img_info
    h = h5py.File('data/gqa_objects.hdf5', 'r')
    img = h['features']
    img_info = json.load(open('data/gqa_objects_merged_info.json', 'r'))
    return img, img_info


class GQA(Dataset):
    def __init__(self, root, split='train'):
        with open(f'data/gqa_all_{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.root = root
        self.split = split
        # self.transform = transform
        self.img, self.img_info = gqa_obj_feature_loader()

    def __getitem__(self, index):
        imgid, question, answer = self.data[index]
        loq = len(question)
        idx = int(self.img_info[imgid]['index'])
        img_obj_features = torch.from_numpy(self.img[idx])
        # question = torch.tensor(question, dtype=torch.long)
        # idx = int(self.img_info[imgid]['index'])
        # img_spt_features = torch
        return img_obj_features, question, loq, answer
            # 100*2048 features.

    def __len__(self):
        return len(self.data)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


def collate_data(batch):
    images, lengths, answers = [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers)


if __name__ == '__main__':
    with open(f'data/gqa_train_balanced.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data[0])

