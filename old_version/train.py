from tqdm import tqdm
import os
import time
import torch
import torch.nn as nn
from dataset import GQA, transform, collate_data
from torch.utils.data import DataLoader
import glove_embedding as ge
from model import MyModel
from torch.autograd import Variable
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('cuda is available')


def train(num_epochs, batch_size):
    current_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    logger = utils.Logger(os.path.join('out', 'log_'+current_time+'.txt'))
    # logger.write(f'{current_time}logger_batch_size{batch_size}')
    vocab_size = ge.vocab_size
    answer_size = ge.answer_size
    train_set = GQA(root='data', split='train', transform=transform)
    val_set = GQA(root='data', split='val', transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_data, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_data, drop_last=True)
    # add a tqdm bar
    d = iter(train_loader)
    pbar = tqdm(d)
    net = MyModel(out_features=answer_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss = nn.CrossEntropyLoss()

    def _validation(val_loader):
        acc_sum, n = 0.0, 0
        for img, q, loq, a in val_loader:
            img, q, a = (
                img.to(device),
                q.to(device),
                a.to(device),
            )
            a_hat = net(img, q)
            acc_sum += (a_hat.detach().argmax(dim=1) == a).sum().item()
            n += a.shape[0]
        if n <= 0:
            print('validation loader does not work')
            return 0.0
        return acc_sum / n

    for i in range(num_epochs):
        loss_sum = 0.0
        acc_sum = 0.0
        n = 0
        for img, q, loq, a in train_loader:
            img, q, a = (
                img.to(device),
                q.to(device),
                a.to(device),
            )
            net.zero_grad()
            a_hat = net(img, q)
            l = loss(a_hat, a)
            l.backward()
            optimizer.step()
            loss_sum += l.item()
            acc_sum += (a_hat.detach().argmax(dim=1) == a).sum().item()
            n += a.shape[0]
            if n % 64000 == 0:
                print('-1000-', end='')
        vali_acc = _validation(val_loader)
        logger.write('Epoch #%d, Loss:%.3f, Train Acc: %.4f, Validation Acc:%.4f'%(i,loss_sum,acc_sum/n,vali_acc))
        print('Epoch #%d, Loss:%.3f, Train Acc: %.4f, Validation Acc:%.4f'%(i,loss_sum,acc_sum/n,vali_acc))
        try:
            torch.save(net.state_dict(), 'checkpoint/checkpoint_{}.pth'.format(i))
        except:
            print('can not save checkpoint.')
    torch.save(net.state_dict(), 'checkpoint/model_1.pth')


if __name__ == '__main__':
    train(5, 64)


