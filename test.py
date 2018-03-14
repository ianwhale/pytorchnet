# test.py

import time
import torch
from torch.autograd import Variable
import plugins


class Tester:
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation
        self.save_results = args.save_results

        self.env = args.env
        self.port = args.port
        self.dir_save = args.save_dir
        self.log_type = args.log_type

        self.cuda = args.cuda
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size

        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

        # for classification
        self.labels = torch.zeros(self.batch_size).long()
        self.inputs = torch.zeros(
            self.batch_size,
            self.resolution_high,
            self.resolution_wide
        )

        if args.cuda:
            self.labels = self.labels.cuda()
            self.inputs = self.inputs.cuda()

        self.inputs = Variable(self.inputs, volatile=True)
        self.labels = Variable(self.labels, volatile=True)

        # logging testing
<<<<<<< HEAD
        # self.log_loss_test = plugins.Logger(args.logs, 'TestLogger.txt')
        self.params_loss_test = ['Loss', 'Accuracy']
        # self.log_loss_test.register(self.params_loss_test)
=======
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger.txt',
            self.save_results
        )
        self.params_loss = ['Loss', 'Accuracy']
        self.log_loss.register(self.params_loss)
>>>>>>> original/master

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Loss': {'dtype': 'running_mean'},
            'Accuracy': {'dtype': 'running_mean'}
        }
        self.monitor.register(self.params_monitor)

        # visualize testing
<<<<<<< HEAD
        # self.visualizer_test = plugins.Visualizer(self.port, 'Test')
        # self.params_visualizer_test = {
        #     'Loss': {'dtype': 'scalar', 'vtype': 'plot'},
        #     'Accuracy': {'dtype': 'scalar', 'vtype': 'plot'},
        #     'Image': {'dtype': 'image', 'vtype': 'image'},
        #     'Images': {'dtype': 'images', 'vtype': 'images'},
        # }
        # self.visualizer_test.register(self.params_visualizer_test)

        # display testing progress
        self.print_test = 'Test [%d/%d][%d/%d] '
        for item in self.params_loss_test:
            self.print_test = self.print_test + item + " %.4f "
=======
        self.visualizer = plugins.Visualizer(self.port, self.env, 'Test')
        self.params_visualizer = {
            'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss',
                     'layout': {'windows': ['train', 'test'], 'id': 1}},
            'Accuracy': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'accuracy',
                         'layout': {'windows': ['train', 'test'], 'id': 1}},
            'Test_Image': {'dtype': 'image', 'vtype': 'image',
                           'win': 'test_image'},
            'Test_Images': {'dtype': 'images', 'vtype': 'images',
                            'win': 'test_images'},
        }
        self.visualizer.register(self.params_visualizer)

        if self.log_type == 'traditional':
            # display training progress
            self.print_formatter = 'Test [%d/%d][%d/%d] '
            for item in self.params_loss:
                self.print_formatter += item + " %.4f "
        elif self.log_type == 'progressbar':
            # progress bar message formatter
            self.print_formatter = '({}/{})' \
                                   ' Load: {:.6f}s' \
                                   ' | Process: {:.3f}s' \
                                   ' | Total: {:}' \
                                   ' | ETA: {:}'
            for item in self.params_loss:
                self.print_formatter += ' | ' + item + ' {:.4f}'
>>>>>>> original/master

        self.evalmodules = []
        self.losses = {}

    def model_eval(self):
        self.model.eval()

    def test(self, epoch, dataloader):
        dataloader = dataloader['test']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()

<<<<<<< HEAD
        i = 0
        acc_sum = 0
        while i < len(dataloader):
=======
        if self.log_type == 'progressbar':
            # progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<10}'.format('Test'), max=len(dataloader))
        end = time.time()

        for i, (inputs, labels) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end
>>>>>>> original/master

            ############################
            # Evaluate Network
            ############################

            batch_size = inputs.size(0)
            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(labels.size()).copy_(labels)

            self.model.zero_grad()
<<<<<<< HEAD
            output = self.model(self.input)
            loss = self.criterion(output, self.label)

            # this is for classification
            pred = output.data.max(1)[1]
            acc = pred.eq(self.label.data).cpu().sum() * 100 / batch_size
            acc_sum = acc_sum + acc
            self.losses_test['Accuracy'] = acc
            self.losses_test['Loss'] = loss.data[0]
            self.monitor_test.update(self.losses_test, batch_size)

            # print batch progress
            # print(self.print_test % tuple(
            #     [epoch, self.nepochs, i, len(dataloader)] +
            #     [self.losses_test[key] for key in self.params_monitor_test]))

        loss = self.monitor_test.getvalues()
        # self.log_loss_test.update(loss)
        loss['Image'] = input[0]
        loss['Images'] = input
        acc_avg = acc_sum/len(dataloader)
        # self.visualizer_test.update(loss)
        return self.monitor_test.getvalues('Loss'),acc_avg
=======
            output = self.model(self.inputs)
            loss = self.criterion(output, self.labels)

            acc = self.evaluation(output, self.labels)

            self.losses['Accuracy'] = acc
            self.losses['Loss'] = loss.data[0]
            self.monitor.update(self.losses, batch_size)

            if self.log_type == 'traditional':
                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i, len(dataloader)] +
                    [self.losses[key] for key in self.params_monitor]))
            elif self.log_type == 'progressbar':
                # update progress bar
                batch_time = time.time() - end
                processed_data_len += len(inputs)
                bar.suffix = self.print_formatter.format(
                    *[processed_data_len, len(dataloader.sampler), data_time,
                      batch_time, bar.elapsed_td, bar.eta_td] +
                     [self.losses[key] for key in self.params_monitor]
                )
                bar.next()
                end = time.time()

        if self.log_type == 'progressbar':
            bar.finish()

        loss = self.monitor.getvalues()
        self.log_loss.update(loss)
        loss['Test_Image'] = inputs[0]
        loss['Test_Images'] = inputs
        self.visualizer.update(loss)
        return self.monitor.getvalues('Loss')
>>>>>>> original/master
