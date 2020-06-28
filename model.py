# Author: yifanh1
# Data: 6/15/2020
# Time: 2:31 PM

# model 1: use vgg 16 to get image features
# input: original images and token questions/answers
import torch
from torch import nn
import glove_embedding
from torch.autograd import Variable
import math
from attention import Linear
# from torchvision import models


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})  # matrix : tensor
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


class Qmodel(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, out_feature):
        super(Qmodel, self).__init__()
        self.embedding, vocab_size, embedding_dim = create_emb_layer(weights_matrix, True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense1 = nn.Linear(hidden_size, out_feature)
        self.dropout = nn.Dropout(0.2)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        y = self.dense1(ht[-1])
        return y

    def init_hidden_layer(self, batch_size):
        weight = next(self.parameters()).data
        # next() retrieve the next item from the iterator,
        # here, weight is the first parameter in the net
        hidden_shape = (self.num_layers, batch_size, self.hidden_size)
        return (Variable(weight.new(*hidden_shape).zero_()),
                Variable(weight.new(*hidden_shape).zero_()))
        # variable.zero_() make the variable to be 0
        # Variable(weight.new(shape1,shape2,shape3) is a variable that
        # has the same dtype with valuable weight


# NLP Module: V 1.1.0
NLP_out_feature = 2048
embedding_dim = 300
hidden_size = 2048
weights_matrix = torch.tensor(glove_embedding.get_embedding_weights())
netNLP = Qmodel(weights_matrix=weights_matrix, hidden_size=hidden_size, num_layers=2, out_feature=NLP_out_feature)

# CNN Module: V 1.2.0
CNN_out_feature = 2048


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CNNModel(nn.Module):
    def __init__(self, features):
        super(CNNModel, self).__init__()
        self.features = features
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.linear = Linear(7*7*256, 2048)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


netCNN = CNNModel(make_layers([64, 'M', 128, 'M', 256, 256, 'M'], batch_norm=True))


# Classifier: V 1.0.0
class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('linear1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('linear2', nn.Linear(mid_features, out_features))


in_features = CNN_out_feature
mid_features = 3000


# Element-wise multiplication and dense layers
class MyModel(nn.Module):
    def __init__(self, out_features):
        super(MyModel, self).__init__()
        self.pointwise_features = CNN_out_feature
        self.out_features = out_features
        self.text = netNLP
        self.cnn = netCNN
        self.dense = nn.Linear(self.pointwise_features, in_features)
        self.classifier = Classifier(in_features, mid_features, out_features, drop=0.5)

    def forward(self, Image, Question):
        Y = self.cnn(Image)
        Z = self.text(Question)
        Z = Y * Z
        Z = self.dense(Z)
        Z = self.classifier(Z)
        return Z


if __name__ == '__main__':
    img = torch.rand((4,3,224,224))
    # print(img)
    # y = new_vgg(img)
    # print(y.shape)
    # print(netCNN)
    # print(netCNN(img))
    print(netCNN(img).shape)

