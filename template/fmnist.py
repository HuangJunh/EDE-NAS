"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import load_FMNIST as data_loader
import os
from datetime import datetime
import multiprocessing
from utils import Utils

class StdConv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride=1):
        super(StdConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size, stride, padding=int((kernel_size - 1) / 2), bias=False),
            nn.BatchNorm2d(planes, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.op(x)
        return out

class DW_SepConv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride=1):
        super(DW_SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding=int((kernel_size - 1) / 2), bias=False, groups=in_planes),
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(planes, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.op(x)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #generated_init


    def forward(self, x):
        #generate_forward

        out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = F.relu(self.conv_end1(out))
        out = out.view(out.size(0), -1)
        # out = self.Hswish(self.dropout(self.linear1(out)))
        # out = F.dropout(out, p=0.2, training=self.training)
        out = self.linear(out)
        return out


class TrainModel(object):
    def __init__(self, is_test):
        if is_test:
            full_trainloader = data_loader.get_train_loader('./datasets/Fashion_MNIST_data', batch_size=128, augment=True,shuffle=True, random_seed=2312391, show_sample=False,num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader('./datasets/Fashion_MNIST_data', batch_size=128, shuffle=False,num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader
        else:
            trainloader, validate_loader = data_loader.get_train_valid_loader('./datasets/Fashion_MNIST_data', batch_size=128,augment=True, valid_size=0.1, shuffle=True,random_seed=2312390, show_sample=False,num_workers=4, pin_memory=True)
            self.trainloader = trainloader
            self.validate_loader = validate_loader
        net = EvoCNNModel()
        cudnn.benchmark = True
        net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        self.net = net
        self.criterion = criterion.cuda()
        self.best_acc = best_acc
        self.best_epoch = 0
        # self.trainloader = trainloader
        # self.validate_loader = validate_loader
        # self.testloader = testloader
        self.file_id = os.path.basename(__file__).split('.')[0]
        #self.testloader = testloader
        #self.log_record(net, first_time=True)
        #self.log_record('+'*50, first_time=False)

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f'% (epoch+1, running_loss/total, (correct/total)))

    def final_train(self, epoch, optimizer):
        self.net.train()
        if epoch ==0: lr = 0.01
        if epoch > 0: lr = 0.1
        if epoch > 30: lr = 0.01
        if epoch > 60: lr = 0.001
        if epoch > 90: lr = 0.0001
        optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum = 0.9, weight_decay=1e-3)
        running_loss = 0.0
        total = 0
        correct = 0
        for ii, data in enumerate(self.full_trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f' % (epoch + 1, running_loss / total, (correct / total)))

    def validate(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = correct / total
        self.log_record('Validate-Loss:%.4f, Acc:%.4f'%(test_loss/total, correct/total))

    def process(self):
        total_epoch = Utils.get_params('network', 'epoch_test')
        min_epoch_eval = Utils.get_params('network', 'min_epoch_eval')
        lr_rate = 0.05
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr_rate, momentum=0.9, weight_decay=1e-3, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, min_epoch_eval)
        is_terminate = 0
        params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.log_record('#parameters:%d' % (params))
        for p in range(min_epoch_eval):
            self.train(p, optimizer)
            scheduler.step()
            self.validate(p)
        return self.best_acc, params

    def process_test(self):
        params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.log_record('#parameters:%d' % (params))
        total_epoch = Utils.get_params('network', 'epoch_test')
        # lr_rate = 0.05
        # optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=1e-3, nesterov=True)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
        for p in range(total_epoch):
            self.final_train(p, None)
            self.test(p)
            # scheduler.step()
        return self.best_acc, params

    def test(self,p):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            self.best_acc = correct / total
        self.log_record('Test-Loss:%.4f, Acc:%.4f' % (test_loss / total, correct / total))

class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, return_dict):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        params = 1e9
        try:
            m = TrainModel(is_test)
            m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            if is_test:
                best_acc, params = m.process_test()
            else:
                best_acc, params = m.process()
            # return_dict[file_id] = best_acc
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.4f'%best_acc)

            f = open('./populations/after_%02d.txt'%(curr_gen), 'a+')
            f.write('%s=%.5f\n'%(file_id, best_acc))
            f.flush()
            f.close()

            f = open('./populations/params_%02d.txt' % (curr_gen), 'a+')
            f.write('%s=%d\n' % (file_id, params))
            f.flush()
            f.close()
"""


