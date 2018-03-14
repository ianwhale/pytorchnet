# config.py
import os
import datetime
import argparse
import json
import configparser
import utils
import re
from ast import literal_eval as make_tuple

<<<<<<< HEAD
result_path = "results/"
result_path = os.path.join(result_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

parser = argparse.ArgumentParser(description='Your project title goes here')
add_arg = parser.add_argument

# ======================== Data Setings ============================================
add_arg('--dataset-test', type=str, default='CIFAR10', help='name of training dataset')
add_arg('--dataset-train', type=str, default='CIFAR10', help='name of training dataset')
add_arg('--split_test', type=float, default=None, help='test split')
add_arg('--split_train', type=float, default=None, help='train split')
add_arg('--dataroot', type=str, default='../', help='path to the data')
add_arg('--save', type=str, default=result_path +'Save', help='save the trained models here')
add_arg('--logs', type=str, default=result_path +'Logs', help='save the training log files here')
add_arg('--resume', type=str, default=None, help='full path of models to resume training')
add_arg('--nclasses', type=int, default=10, help='number of classes for classification')
add_arg('--input-filename-test', type=str, default=None, help='input test filename for filelist and folderlist')
add_arg('--label-filename-test', type=str, default=None, help='label test filename for filelist and folderlist')
add_arg('--input-filename-train', type=str, default=None, help='input train filename for filelist and folderlist')
add_arg('--label-filename-train', type=str, default=None, help='label train filename for filelist and folderlist')
add_arg('--loader-input', type=str, default=None, help='input loader')
add_arg('--loader-label', type=str, default=None, help='label loader')

# ======================== Network Model Setings ===================================
add_arg('--nchannels', type=int, default=3, help='number of input channels')
add_arg('--nfilters', type=int, default=64, help='number of filters in conv layer')
add_arg('--resolution-high', type=int, default=32, help='image resolution height')
add_arg('--resolution-wide', type=int, default=32, help='image resolution width')
add_arg('--ndim', type=int, default=None, help='number of feature dimensions')
add_arg('--nunits', type=int, default=None, help='number of units in hidden layers')
add_arg('--dropout', type=float, default=None, help='dropout parameter')
add_arg('--net-type', type=str, default=None, help='type of network')
add_arg('--length-scale', type=float, default=None, help='length scale')
add_arg('--tau', type=float, default=None, help='Tau')
add_arg('--genome_id', type=int, default=None, metavar='', help='none')

# ======================== Training Settings =======================================
add_arg('--cuda', type=bool, default=True, help='run on gpu')
add_arg('--ngpu', type=int, default=1, help='number of gpus to use')
add_arg('--batch-size', type=int, default=64, help='batch size for training')
add_arg('--nepochs', type=int, default=100, help='number of epochs to train')
add_arg('--niters', type=int, default=None, help='number of iterations at test time')
add_arg('--epoch-number', type=int, default=None, help='epoch number')
add_arg('--nthreads', type=int, default=5, help='number of threads for data loading')
add_arg('--manual-seed', type=int, default=0, help='manual seed for randomness')
add_arg('--port', type=int, default=None, help='port for visualizing training at http://localhost:port')

# ======================== Hyperparameter Setings ==================================
add_arg('--optim-method', type=str, default='Adam', help='the optimization routine ')
add_arg('--learning-rate', type=float, default=3e-4, help='learning rate')
add_arg('--learning-rate-decay', type=float, default=None, help='learning rate decay')
add_arg('--momentum', type=float, default=0.9, help='momentum')
add_arg('--weight-decay', type=float, default=0.0, help='weight decay')
add_arg('--adam-beta1', type=float, default=0.9, help='Beta 1 parameter for Adam')
add_arg('--adam-beta2', type=float, default=0.999, help='Beta 2 parameter for Adam')

args = parser.parse_args()
=======

def parse_args():
    result_path = "results/"
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = os.path.join(result_path, now)

    parser = argparse.ArgumentParser(description='Your project title goes here')

    # the following two parameters can only be provided at the command line.
    parser.add_argument('--result-path', type=str, default=result_path, metavar='', help='full path to store the results')
    parser.add_argument("-c", "--config", "--args-file", dest="config_file", default="args.txt", help="Specify a config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()

    result_path = args.result_path
    # add date and time to the result directory name
    if now not in result_path:
        result_path = os.path.join(result_path, now)

    # ======================= Data Setings =====================================
    parser.add_argument('--dataset-root-test', type=str, default=None, help='path of the data')
    parser.add_argument('--dataset-root-train', type=str, default=None, help='path of the data')
    parser.add_argument('--dataset-test', type=str, default=None, help='name of training dataset')
    parser.add_argument('--dataset-train', type=str, default=None, help='name of training dataset')
    parser.add_argument('--split_test', type=float, default=None, help='test split')
    parser.add_argument('--split_train', type=float, default=None, help='train split')
    parser.add_argument('--test-dev-percent', type=float, default=None, metavar='', help='percentage of dev in test')
    parser.add_argument('--train-dev-percent', type=float, default=None, metavar='', help='percentage of dev in train')
    parser.add_argument('--save-dir', type=str, default=os.path.join(result_path, 'Save'), metavar='', help='save the trained models here')
    parser.add_argument('--logs-dir', type=str, default=os.path.join(result_path, 'Logs'), metavar='', help='save the training log files here')
    parser.add_argument('--resume', type=str, default=None, help='full path of models to resume training')
    parser.add_argument('--nclasses', type=int, default=None, metavar='', dest='noutputs', help='number of classes for classification')
    parser.add_argument('--noutputs', type=int, default=None, metavar='', help='number of outputs, i.e. number of classes for classification')
    parser.add_argument('--input-filename-test', type=str, default=None, help='input test filename for filelist and folderlist')
    parser.add_argument('--label-filename-test', type=str, default=None, help='label test filename for filelist and folderlist')
    parser.add_argument('--input-filename-train', type=str, default=None, help='input train filename for filelist and folderlist')
    parser.add_argument('--label-filename-train', type=str, default=None, help='label train filename for filelist and folderlist')
    parser.add_argument('--loader-input', type=str, default=None, help='input loader')
    parser.add_argument('--loader-label', type=str, default=None, help='label loader')
    parser.add_argument('--dataset-options', type=json.loads, default=None, metavar='', help='additional model-specific parameters, i.e. \'{"gauss": 1}\'')

    # ======================= Network Model Setings ============================
    parser.add_argument('--model-type', type=str, default=None, help='type of network')
    parser.add_argument('--model-options', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--loss-type', type=str, default=None, help='loss method')
    parser.add_argument('--loss-options', type=json.loads, default={}, metavar='', help='loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--evaluation-type', type=str, default=None, help='evaluation method')
    parser.add_argument('--evaluation-options', type=json.loads, default={}, metavar='', help='evaluation-specific parameters, i.e. \'{"topk": 1}\'')
    parser.add_argument('--resolution-high', type=int, default=None, help='image resolution height')
    parser.add_argument('--resolution-wide', type=int, default=None, help='image resolution width')
    parser.add_argument('--ndim', type=int, default=None, help='number of feature dimensions')
    parser.add_argument('--nunits', type=int, default=None, help='number of units in hidden layers')
    parser.add_argument('--dropout', type=float, default=None, help='dropout parameter')
    parser.add_argument('--length-scale', type=float, default=None, help='length scale')
    parser.add_argument('--tau', type=float, default=None, help='Tau')

    # ======================= Training Settings ================================
    parser.add_argument('--cuda', type=utils.str2bool, default=None, help='run on gpu')
    parser.add_argument('--ngpu', type=int, default=None, help='number of gpus to use')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size for training')
    parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--niters', type=int, default=None, help='number of iterations at test time')
    parser.add_argument('--epoch-number', type=int, default=None, help='epoch number')
    parser.add_argument('--nthreads', type=int, default=None, help='number of threads for data loading')
    parser.add_argument('--manual-seed', type=int, default=None, help='manual seed for randomness')

    # ===================== Visualization Settings =============================
    parser.add_argument('-p', '--port', type=int, default=None, metavar='', help='port for visualizing training at http://localhost:port')
    parser.add_argument('--env', type=str, default='', metavar='', help='environment for visualizing training at http://localhost:port')

    # ======================= Hyperparameter Setings ===========================
    parser.add_argument('--learning-rate', type=float, default=None, help='learning rate')
    parser.add_argument('--optim-method', type=str, default=None, help='the optimization routine ')
    parser.add_argument('--optim-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters, i.e. \'{"lr": 0.001}\'')
    parser.add_argument('--scheduler-method', type=str, default=None, help='cosine, step, exponential, plateau')
    parser.add_argument('--scheduler-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters')

    # ======================== Main Setings ====================================
    parser.add_argument('--log-type', type=str, default='traditional', metavar='', help='allows to select logger type, traditional or progressbar')
    parser.add_argument('--same-env', type=utils.str2bool, default='No', metavar='', help='does not add date and time to the visdom environment name')
    parser.add_argument('-s', '--save', '--save-results', type=utils.str2bool, dest="save_results", default='No', metavar='', help='save the arguments and the results')

    if os.path.exists(args.config_file):
        config = configparser.ConfigParser()
        config.read([args.config_file])
        defaults = dict(config.items("Arguments"))
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)

    # add date and time to the name of Visdom environment and the result
    if args.env is '':
        args.env = args.model_type
    if not args.same_env:
        args.env += '_' + now
    args.result_path = result_path

    # refine tuple arguments: this section converts tuples that are
    #                         passed as string back to actual tuples.
    pattern = re.compile('^\(.+\)')

    for arg_name in vars(args):
        # print(arg, getattr(args, arg))
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str) and pattern.match(arg_value):
            setattr(args, arg_name, make_tuple(arg_value))
            print(arg_name, arg_value)
        elif isinstance(arg_value, dict):
            dict_changed = False
            for key, value in arg_value.items():
                if isinstance(value, str) and pattern.match(value):
                    dict_changed = True
                    arg_value[key] = make_tuple(value)
            if dict_changed:
                setattr(args, arg_name, arg_value)

    return args
>>>>>>> original/master
