# main.py

import torch
import random
import numpy as np
import os
import utils
from model import Model
from test import Tester
from train import Trainer
from config import parser
from dataloader import Dataloader
import time
import pickle
from checkpoints import Checkpoints


def main():
    # parse the arguments
    args = parser.parse_args()
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    # utils.saveargs(args)

    # initialize the checkpoint class
    # checkpoints = Checkpoints(args)


    # Create Model
    models = Model(args)
    model, criterion = models.setup()
    # print(model)

    # Data Loading
    dataloader = Dataloader(args)
    loaders = dataloader.create()

    # The trainer handles the training loop
    trainer = Trainer(args, model, criterion)
    # The trainer handles the evaluation on validation set
    tester = Tester(args, model, criterion)

    # start training !!!
    loss_best = 1e10
    acc_test_list = []
    inference_time_list = []
    acc_best = 0
    for epoch in range(args.nepochs):

        # train for a single epoch
        start_time_epoch = time.time()
        loss_train, acc_train = trainer.train(epoch, loaders)
        inference_time_start = time.time()
        loss_test, acc_test = tester.test(epoch, loaders)
        inference_time_list.append(np.round((time.time() - inference_time_start), 2))
        acc_test_list.append(acc_test)
        # if loss_best > loss_test:
        #     model_best = True
        #     loss_best = loss_test
        #     checkpoints.save(epoch, model, model_best)

        time_elapsed = np.round((time.time() - start_time_epoch), 2)

        # update the best test accu found so found
        if acc_test > acc_best:
            acc_best = acc_test

        print("Epoch {}, train loss = {}, test accu = {}, best accu = {}, {} sec"
              .format(epoch, np.average(loss_train), acc_test, acc_best, time_elapsed))

    # save the final model parameter
    # torch.save(model.state_dict(),
    #            "model_file/model%d.pth" % int(args.genome_id - 1))
    accuracy = np.mean(acc_test_list[-5:])
    inference_time = np.median(inference_time_list)

    # accuracy = acc_best
    fitness = [accuracy, inference_time]
    with open("output_file/output%d.pkl" % int(args.genome_id - 1), "wb") as f:
        pickle.dump(fitness, f)

if __name__ == "__main__":
    main()
