# evohourglass.py

import torch
import torch.nn as nn
from evolution import HourGlassResidual, LOSHourGlassDecoder


def get_decoder(decoder_str, genome, n_stacks, out_channels):
    """
    Construct the appropriate decoder.
    :param decoder_str: string, refers to what genome scheme we're using.
    :param genome: list, the genome.
    :param n_stacks: int, number of hourglasses to use.
    :param out_channels: int, number of output feature maps.
    :return: evolution.HourGlassDecoder
    """
    if decoder_str == "los":  # "Line of sight" decoder.
        return LOSHourGlassDecoder(genome, n_stacks, out_channels)

    raise NotImplementedError("Decoder {} not implemented.".format(decoder_str))


class EvoHourGlass(nn.Module):
    """
    Evolutionary hourglass network.
    Repeats the evolved block.
    """

    def __init__(self, genome, n_stacks, out_channels, decoder="los"):
        """
        EvoHourGlass constructor.
        :param genome: list, list of ints, represents the genome.
        :param n_stacks: int, number of hourglasses to stack.
        :param out_channels: int, number of outputs feature maps.
        :param decoder: string, type of decoder to use.
        """
        super(EvoHourGlass, self).__init__()

        self.model = get_decoder(decoder, genome, n_stacks, out_channels).get_model()

    def forward(self, x):
        return self.model(x)


def demo():
    from plugins import make_dot_backprop

    data = torch.randn(1, 3, 256, 256)
    genome = [1, 2, 3, 2, 1, 2, 3, 2, 1]  # W network.
    model = EvoHourGlass(genome, n_stacks=1, out_channels=16, decoder="los")

    out = model(torch.autograd.Variable(data))
    make_dot_backprop(out[-1]).view()



if __name__ == "__main__":
    demo()
