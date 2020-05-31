#!/usr/bin/env python
# coding: utf-8

# read the json file and define a Q&A generator
import ijson
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from PIL import Image
# use GloVe embedding layer:
# make a dict of GLOVE vectors
import os

embedding_dim = 100
max_len = 20
# from torchtext.vocab import GloVe

glove_dir = 'glove/'
glove_path = 'glove.6B.100d.txt'
f = open(os.path.join(glove_dir, glove_path), 'r', encoding='UTF-8')
glove = {}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove[word] = coefs
f.close()
# print('Found %d words in glove vectors'%(len(glove)))

voc = ['Unknown'] + list(glove.keys())[:10000]
voc_len = len(voc)
words_found = 0
weights_matrix = np.zeros((voc_len, embedding_dim))
for i, word in enumerate(voc):
    try:
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))


# print(words_found)
# Create pretrained embedding layer:
# glove=GloVe(name='6B',dim = embedding_dim)#How to use? glove['a']
weights_matrix = torch.from_numpy(weights_matrix)


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings = voc_len
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})  # matrix : tensor
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


# embedding_layer, x, xy= create_emb_layer(weights_matrix)

def text2tensor(string):
    string = string.lower()
    words = string.split()
    tokens = []
    for word in words:
        try:
            word = ''.join([c for c in word if c.isalpha()])
            index = voc.index(word)
            tokens.append(index)
        except ValueError:
            print('can not find', word, end=' ')
            wor = word[:-1]
            try:
                index = voc.index(wor)
                tokens.append(index)
            except ValueError:
                tokens.append(0)
                print('can not find', wor)
    tokens = np.array(tokens, dtype='int')
    # print(tokens)
    return tokens


def questiongenerator(train_path, batchsize):  # give the image no. and question & answer
    parser = ijson.parse(open(train_path, 'r', encoding='utf-8'))
    # load json iteratively
    flag = 0
    q = np.zeros((batchsize, max_len), dtype='long')
    a = np.zeros((batchsize, 1), dtype='long')
    imgId = []
    for prefix, event, value in parser:
        if prefix.endswith('.question') and value is not None and event == 'string':
            flag += 1
            value = text2tensor(value)
            q[flag - 1][:len(value)] = value
        elif prefix.endswith('.imageId') and value is not None:
            imgId.append(value)
        elif prefix.endswith('.answer') and value is not None and event == 'string':
            value = text2tensor(value)
            a[flag - 1] = value
            if flag >= batchsize:
                flag = 0
                q = torch.tensor(q, dtype=torch.long)
                a = torch.tensor(a, dtype=torch.long)
                yield (imgId, q, a)
                q = np.zeros((batchsize, max_len), dtype='long')
                a = np.zeros((batchsize, 1), dtype='long')
                imgId = []


# Process images:
# imshow(torchvision.utils.make_grid(imgs))
# define an image generator


def imagefinder(image_train_path, img_no):
    def _find_image(image_train_path, no):
        # img = img/2 + 0.5
        # npimg = img.numpy()
        # plt.imshow(np.transpose(npimg,(1,2,0)))
        # plt.show()
        transform = transforms.Compose([
            transforms.CenterCrop((600, 800)),  # h x w
            transforms.Resize((150, 200)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = Image.open(image_train_path + no + '.jpg')
        img = transform(img)
        img = img.numpy()
        return img

    ''' for img_no, question, answer in questiongenerator(text_train_path):
        img_no = int(img_no)
        img = _find_image(image_train_path, img_no)'''
    imgs = []
    for i in range(len(img_no)):
        imgs.append(_find_image(image_train_path, img_no[i]))
    imgs = torch.Tensor(imgs)

    return imgs


if __name__ == '__main__':
    train_path = 'text/train_balanced_questions.json'
    batchsize = 2
    for imgid, q, a in questiongenerator(train_path,batchsize):
        print(q)
        print(a)
        break
