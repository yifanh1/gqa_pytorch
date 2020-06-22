import torch
from torch import nn
import glove_embedding
from torch.autograd import Variable


def init_weights(net, std=0.01):
    net.weight.data.normal_(0.0, std)
    return net


class FC(nn.Module):
    def __init__(self, dims):
        # dims: list of dims: [dim_input, dim_mid1, dim_mid2, ..., dim_out]
        super(FC, self).__init__()
        layers = []
        for i in range(len(dims)-1):
            dim_input = dims[i]
            dim_output = dims[i+1]
            linear = init_weights(nn.Linear(dim_input, dim_output))
            layers.append(linear)
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Linear(nn.Module):
    def __init__(self, input_dim, out_dim, init_std=0.01):
        super(Linear, self).__init__()
        self.std = init_std
        self.linear = init_weights(nn.Linear(input_dim, out_dim), init_std)

    def forward(self, x):
        return self.linear(x)


class Attention(nn.Module):
    def __init__(self, self_attention=True, q_dim=2048, v_dim=2048, hid_dim=1024):
        super(Attention, self).__init__()
        self.nonlinear = FC([v_dim + q_dim, hid_dim])
        self.linear = Linear(hid_dim,1)
        self.sa = self_attention

    def forward(self, v, q):
        '''
        :param v: img features, shape=[batch_size, num_objs, v_dim(default=2048)]
        :param q: question features, shape=[batch_size, q_dim(default=300)]
        :return: weights w
        '''
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)  # fit 2 shape
        vq = torch.cat((v, q), 2)  # vq shape: [bsz, num_objs, v_dim+q_dim]
        # calculate a = w ReLU(Wv+Wa)
        a_vector = self.nonlinear(vq) # bsz, num_objs, hid_dim
        a_vector = self.linear(a_vector) # bsz, num_objs, 1
        # alpha = softmax(a)
        alpha = nn.functional.softmax(a_vector, dim=1)  # bsz, num_objs, 1
        return alpha  # alpha is the weights


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})  # matrix : tensor
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


class Qmodel(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, out_feature, self_attention=False):
        super(Qmodel, self).__init__()
        self.embedding, vocab_size, embedding_dim = create_emb_layer(weights_matrix, True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense1 = nn.Linear(hidden_size, out_feature)
        self.dropout = nn.Dropout(0.2)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.nonlinear = FC([hidden_size, 1024])  # ht[:-1] size: num_layers-1, bsz, hidden_size
        self.linear = Linear(1024, 1)
        self.sa = self_attention

    def attention(self, h):  # len_seq, bsz, hidden_size
        h = h.permute(1, 0, 2)
        a_vector = self.nonlinear(h)  # bsz, len_seq, hid
        a_vector = self.linear(a_vector)  # bsz, len_seq, 1
        # alpha = softmax(a)
        alpha = nn.functional.softmax(a_vector, dim=1)  # bsz, num_objs, 1
        return alpha  # alpha is the weights

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        if self.sa:
            alpha = self.attention(lstm_out)
            x = (alpha * lstm_out).sum(1)
            y = self.dense1(x)
        else:
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

    def hidden_states(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        return lstm_out


# NLP Module: V 1.1.0
NLP_out_feature = 2048
embedding_dim = 300
hidden_size = 2048
# Classifier config
in_features = 2048
mid_features = 3000

weights_matrix = torch.tensor(glove_embedding.get_embedding_weights())
netNLP = Qmodel(weights_matrix=weights_matrix, hidden_size=hidden_size, num_layers=2, out_feature=NLP_out_feature, self_attention=False)

# Attention Module: V 1.0.0
netAttention = Attention(q_dim=2048, v_dim=2048, hid_dim=1024)


# Classifier: V 1.0.0
class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('linear1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('linear2', nn.Linear(mid_features, out_features))


# Element-wise multiplication and dense layers
class MyModel(nn.Module):
    def __init__(self, out_features):
        super(MyModel, self).__init__()
        self.pointwise_features = 2048
        self.out_features = out_features
        self.text = netNLP
        self.attention = netAttention
        self.dense = nn.Linear(self.pointwise_features, in_features)
        self.classifier = Classifier(in_features, mid_features, out_features, drop=0.5)

    def forward(self, v, q):
        y_q = self.text(q)
        alpha = self.attention(v, y_q)  # bsz, num_objs, 1
        y_v = (alpha * v).sum(1)  # bsz, 2048(num_features)
        z = y_v * y_q
        z = self.dense(z)
        Z = self.classifier(z)
        return Z




