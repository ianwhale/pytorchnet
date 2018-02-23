# decoder.py

import torch.nn as nn
from abc import ABC, abstractmethod


class Decoder(ABC):
    """
    Abstract genome decoder class.
    """
    @abstractmethod
    def __init__(self, list_genome):
        """
        :param list_genome: genome represented as a list.
        """
        self._genome = list_genome

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()

    @staticmethod
    def build_layers(phases):
        layers = []
        last_phase = phases.pop()
        for phase in phases:
            layers.append(phase)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # TODO: Generalize this, or consider a new genome.

        layers.append(last_phase)
        return layers

#
# TODO: Implement all these, move all to their own file.
#


class GCNNGenomeDecoder(Decoder):
    """
    Original genetic CNN genome from: https://arxiv.org/abs/1703.01513
    """
    def __init__(self, list_genome):
        super().__init__(list_genome)
        pass

    def get_model(self):
        pass


class DONGenomeDecoder(Decoder):
    """
    'Double Or Nothing genome' decoder.
    DON refers to the channel size strategy which either doubles or does before a phase.
    Also defines residual as ResidualGenome does.
    """
    def __init__(self, list_genome):
        super().__init__(list_genome)
        pass

    def get_model(self):
        pass


class Identity(nn.Module):
    """
    Adding an identity allows us to keep things general in certain places.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x