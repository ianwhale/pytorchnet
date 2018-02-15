# decoder.py

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