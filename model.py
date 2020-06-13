# Author: yifanh1
# Data: 6/10/2020
# Time: 5:24 PM

import torch
from torch import nn
import glove_embedding
from torch.autograd import Variable
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

# CNN Module: V 1.1.0
CNN_out_feature = 2048
state_dict = torch.load('data/vgg16.pth')
# The server can not connect internet, so here is a copy of source code of torchvision.models


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        return x


vgg = vgg16(pretrained=True)
# vgg = models.vgg16(pretrained=True)
# new_vgg = nn.Sequential(*(l[:-1]))
for param in vgg.features[:28].parameters():
    param.requires_grad = False
l = list(vgg.children())[:-1]
vggclassifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, CNN_out_feature),
        )
l = [*l, vggclassifier]
netCNN = nn.Sequential(*l)


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

