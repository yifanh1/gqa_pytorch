#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torchvision.models as models
from preprocess import create_emb_layer, weights_matrix, embedding_dim

# NLP Module: V 1.0.0
NLP_out_feature = 1024


class EmbedLayer(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, out_feature):
        super(EmbedLayer, self).__init__()
        self.embedding, voc_size, embedding_dim = create_emb_layer(weights_matrix, True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense1 = nn.Linear(hidden_size, out_feature)
        # self.dense2 = nn.Linear(voc_size, out_feature)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        # torch.transpose(x, 0, 1)
        lstm_out, (ht, ct) = self.lstm(x)
        y = self.dense1(ht[-1])
        return y


''' 
    def init_hidden_layer(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))'''


netNLP = EmbedLayer(weights_matrix, embedding_dim, 1, NLP_out_feature)

# CNN module:netCNN
CNN_out_feature = 1024

CNN = nn.Sequential()
vgg16 = models.vgg16(pretrained=True)
CNN.add_module('vgg16', vgg16)
CNN.add_module('linear1', nn.Linear(1000, CNN_out_feature))
netCNN = CNN


class Classifier(nn.Sequential):
    def __init__(self, in_features, midfeatures, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('linear1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('linear2', nn.Linear(mid_features, out_features))


in_features = 1024
mid_features = 1024
out_features = 10000
# TODO: config file:
classifier = Classifier(in_features, mid_features, out_features, drop=0.5)


# pointwise multiplication and dense layers
class GQA(nn.Module):
    def __init__(self, out_features):
        super(GQA, self).__init__()
        self.pointwise_features = CNN_out_feature
        self.out_features = out_features
        self.text = netNLP
        self.cnn = CNN
        self.dense = nn.Linear(self.pointwise_features, out_features)
        self.classifier = classifier

    def forward(self, Image, Question):
        Y = self.cnn(Image)
        Z = self.text(Question)
        Z = Y * Z
        Z = self.dense(Z)
        Z = self.classifier(Z)
        return Z


if __name__ == '__main__':
    # print(GQA(1024))
    # print(embedding_dim)
    netNLP = EmbedLayer(weights_matrix, embedding_dim, 1, 5)
    x = torch.zeros((2,20), dtype=torch.long)
    netNLP(x)
