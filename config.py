# config.py
import os
import datetime
import argparse

result_path = "results/"
result_path = os.path.join(result_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

parser = argparse.ArgumentParser(description='Your project title goes here')
add_arg = parser.add_argument

# ======================== Data Setings ============================================
add_arg('--dataset-test', type=str, default=None, help='name of training dataset')
add_arg('--dataset-train', type=str, default=None, help='name of training dataset')
add_arg('--split_test', type=float, default=None, help='test split')
add_arg('--split_train', type=float, default=None, help='train split')
add_arg('--dataroot', type=str, default=None, help='path to the data')
add_arg('--save', type=str, default=result_path +'Save', help='save the trained models here')
add_arg('--logs', type=str, default=result_path +'Logs', help='save the training log files here')
add_arg('--resume', type=str, default=None, help='full path of models to resume training')
add_arg('--nclasses', type=int, default=None, help='number of classes for classification')
add_arg('--input-filename-test', type=str, default=None, help='input test filename for filelist and folderlist')
add_arg('--label-filename-test', type=str, default=None, help='label test filename for filelist and folderlist')
add_arg('--input-filename-train', type=str, default=None, help='input train filename for filelist and folderlist')
add_arg('--label-filename-train', type=str, default=None, help='label train filename for filelist and folderlist')
add_arg('--loader-input', type=str, default=None, help='input loader')
add_arg('--loader-label', type=str, default=None, help='label loader')

# ======================== Network Model Setings ===================================
add_arg('--nchannels', type=int, default=None, help='number of input channels')
add_arg('--nfilters', type=int, default=None, help='number of filters in conv layer')
add_arg('--resolution-high', type=int, default=None, help='image resolution height')
add_arg('--resolution-wide', type=int, default=None, help='image resolution width')
add_arg('--ndim', type=int, default=None, help='number of feature dimensions')
add_arg('--nunits', type=int, default=None, help='number of units in hidden layers')
add_arg('--dropout', type=float, default=None, help='dropout parameter')
add_arg('--net-type', type=str, default=None, help='type of network')
add_arg('--length-scale', type=float, default=None, help='length scale')
add_arg('--tau', type=float, default=None, help='Tau')

# ======================== Training Settings =======================================
add_arg('--cuda', type=bool, default=None, help='run on gpu')
add_arg('--ngpu', type=int, default=None, help='number of gpus to use')
add_arg('--batch-size', type=int, default=None, help='batch size for training')
add_arg('--nepochs', type=int, default=None, help='number of epochs to train')
add_arg('--niters', type=int, default=None, help='number of iterations at test time')
add_arg('--epoch-number', type=int, default=None, help='epoch number')
add_arg('--nthreads', type=int, default=None, help='number of threads for data loading')
add_arg('--manual-seed', type=int, default=None, help='manual seed for randomness')
add_arg('--port', type=int, default=None, help='port for visualizing training at http://localhost:port')

# ======================== Hyperparameter Setings ==================================
add_arg('--optim-method', type=str, default=None, help='the optimization routine ')
add_arg('--learning-rate', type=float, default=None, help='learning rate')
add_arg('--learning-rate-decay', type=float, default=None, help='learning rate decay')
add_arg('--momentum', type=float, default=None, help='momentum')
add_arg('--weight-decay', type=float, default=None, help='weight decay')
add_arg('--adam-beta1', type=float, default=None, help='Beta 1 parameter for Adam')
add_arg('--adam-beta2', type=float, default=None, help='Beta 2 parameter for Adam')

args = parser.parse_args()
