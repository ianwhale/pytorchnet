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
    for epoch in range(args.nepochs):

        # train for a single epoch
        loss_train = trainer.train(epoch, loaders)
        loss_test, acc_test = tester.test(epoch, loaders)
        acc_test_list.append(acc_test)
        # if loss_best > loss_test:
        #     model_best = True
        #     loss_best = loss_test
        #     checkpoints.save(epoch, model, model_best)
        print("Epoch {} Accuracy {}".format(epoch,acc_test))
    accuracy = np.mean(acc_test_list[-5:])
    # else:
    #     # in case no gpu is available
    #
    #     # 1, store the genome infomation
    #     with open("input_file/input%d.pkl" % int(args.genome_id - 1), "rb") as f:
    #         genome = pickle.load(f)
    #     if os.path.exists("error_evaluation"):
    #         pass
    #     else:
    #         os.makedirs("error_evaluation")
    #     filename = os.path.join("error_evaluation",
    #                             (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".pkl"))
    #     with open(filename,"wb") as f:
    #         pickle.dump(genome,f)
    #
    #     #2, artificially assign a bad classification accuracy to it.
    #     accuracy = 0
    with open("output_file/output%d.pkl" % int(args.genome_id - 1), "wb") as f:
        pickle.dump(accuracy, f)

if __name__ == "__main__":
    main()
