# model.py

import math
import models
import losses
import pickle
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Model:

    def __init__(self, args):
        self.ngpu = args.ngpu
        self.cuda = args.cuda
        self.nclasses = args.nclasses
        self.nfilters = args.nfilters
        self.nchannels = args.nchannels
        self.net_type = args.net_type
        self.genome_id = args.genome_id
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

    def setup(self):

        with open("input_file/input%d.pkl" % int(self.genome_id - 1), "rb") as f:
            genome = pickle.load(f)

        # model = getattr(models, self.net_type)(
        #     nchannels=self.nchannels,
        #     nfilters=self.nfilters,
        #     nclasses=self.nclasses,
        # )
        model = models.evonetwork.EvoNetwork(genome,
                                             [(self.nchannels, self.nfilters)],
                                             self.nclasses,(self.resolution_high,
                                                            self.resolution_wide))
        criterion = losses.Classification()

        if self.cuda:
            model = nn.DataParallel(model, device_ids=list(range(self.ngpu)))
            model = model.cuda()
            criterion = criterion.cuda()

        model.apply(weights_init)

        # if checkpoints.latest('resume') is None:
        #     model.apply(weights_init)
        # else:
        #     tmp = checkpoints.load(checkpoints['resume'])
        #     model.load_state_dict(tmp)

        return model, criterion
