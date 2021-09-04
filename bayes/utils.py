""" Utilities """
from torchsampler import ImbalancedDatasetSampler
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import preproc
import np_transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed

class HsdDct(Dataset):
    def __init__(self, data_path, fealen=32, transform=None, input_size=48):
        self.data_path = data_path
        self.transform = transform
        self.fealen = fealen
        self.input_size = input_size
        print("loading data into the main memory...")
        self.ft_buffer, self.label_buffer = self.readcsv(self.data_path, self.fealen)
    
    def readcsv(self, target, fealen=32, n_jobs=-1):
        #read label
        path  = target + '/label.csv'
        label = np.genfromtxt(path, delimiter=',').astype(np.int8)
        #read feature
        
        def _readcsv(i, target):
            if i==0:
                file = '/dc.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).values.astype(np.int8)
                print(i)
                return featemp
            else:
                file = '/ac'+str(i)+'.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).values.astype(np.int8)
                print(i)
                return featemp
        feature = Parallel(n_jobs = n_jobs, prefer="processes")(delayed(_readcsv)(i, target) for i in range(fealen))

        
        return np.rollaxis(np.asarray(feature), 0, 3)[:,:,0:fealen].reshape((-1, self.input_size, self.input_size, fealen)), label

    def __len__(self):
        return len(self.label_buffer)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.ft_buffer[idx].astype(np.float32), int(self.label_buffer[idx])
        if self.transform:
            x = self.transform(x)
        return x, y

import time


def get_data(dataset, data_path, cutout_length, validation, val_path=None, fealen=3, inputsize=48, aug=False):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    elif dataset == 'iccad19':
        dset_cls = HsdDct
        n_classes = 2
    else:
        raise ValueError(dataset)
    
    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length, inputsize)

    if dataset != 'iccad19':
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)
        # assuming shape is NHW or NHWC
        shape = trn_data.train_data.shape
        input_channels = 3 if len(shape) == 4 else 1
        assert shape[1] == shape[2], "not expected shape = {}".format(shape)
        input_size = shape[1]

        ret = [input_size, input_channels, n_classes, trn_data]
        if validation: # append validation data
            ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))
    else:
        trn_data = dset_cls(os.path.join(data_path, 'train'), transform=trn_transform, fealen=fealen, input_size=inputsize)
        input_size = inputsize
        input_channels = fealen
        ret = [input_size, input_channels, n_classes, trn_data]
        if validation: # append validation data
            ret.append(dset_cls(os.path.join(data_path, 'test', val_path), transform=val_transform, fealen=fealen, input_size=inputsize))
            if aug:
                np_transforms.aug_data_binary(ret[3], ret[4])

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SumMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        """ Reset all statistics """
        self.val = 0
    def update(self, val):
        self.val += val

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def binary_conf_mat(output, truth):
    _, prediction = output.max(1)
    conf_mat = confusion_matrix(truth.cpu().numpy(), prediction.cpu().numpy(), labels=[0,1])
    return conf_mat
    
def binary_f1_score(conf_mat):
    if conf_mat[1, 0] + conf_mat[1, 1] > 0:
        recall = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    else:
        recall = 0
    if conf_mat[0, 1] + conf_mat[1, 1] > 0:
        precision = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1])
    else:
        precision = 0
    if precision or recall:
        return 2*(precision*recall)/(precision+recall)
    else:
        return 0

def save_checkpoint(state, ckpt_dir, is_best=False, epoch=0):
    filename = os.path.join(ckpt_dir, '%d_checkpoint.pth.tar'%epoch)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
