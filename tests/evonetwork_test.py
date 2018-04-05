import torch
import unittest
from random import getrandbits
from models.evonetwork import EvoNetwork


def r_bit():
    """
    Get a random bit.
    :return: int, x in {0, 1}.
    """
    return int(getrandbits(1))


class GenomeGenerator:
    """
    This class will generate 'n' random genomes of the form.

    [
        [
            [x_1],
            [x_2, x_3],
            [x_4, x_5, x_6],
            [x_7]
        ],
        [
            [y_1],
            [y_2, y_3],
            [y_4, y_5, y_6],
            [y_7, y_8, y_9, y_10],
            [y_11]
        ]
    ]

    It will be defined as tuples to support hasing, but it will still work.
    """
    def __init__(self, n=2500):
        """
        Constructor.
        :param n:
        """
        self.n = n
        self.i = 0
        self.seen = {}

    def __iter__(self):
        """
        Reset iterator dummy variable and archive.
        """
        self.i = 0
        self.seen = {}

    def __next__(self):
        return self.get_random_genome()

    def get_random_genome(self, encoding="residual"):
        if encoding != "residual":
            raise NotImplementedError("Unsupported encoding.")

        return [
            [
                [r_bit()],
                [r_bit(), r_bit()],
                [r_bit(), r_bit(), r_bit()],
                [r_bit()],
            ],
            [
                [r_bit()],
                [r_bit(), r_bit()],
                [r_bit(), r_bit(), r_bit()],
                [r_bit(), r_bit(), r_bit(), r_bit()],
                [r_bit()],
            ]
        ]


class TestEvoNetwork(unittest.TestCase):
    def test_random_genomes(self):
        """
        Iterate through a bunch of genomes and make sure all of them run without error (best I could think of...).
        This will take a while to run.
        """
        n = 2500

        channels = [(3, 8), (8, 16)]
        out_features = 10
        data = torch.autograd.Variable(torch.randn(16, 3, 32, 32))
        data_shape = (32, 32)
        g = GenomeGenerator()

        while n > 0:
            genome = g.get_random_genome()

            print("Testing: {}".format(genome))

            net = EvoNetwork(genome, channels, out_features, data_shape, decoder="dense")
            net(data)

    def test_special_genomes(self):
        """
        Test a few special case genomes for errors.
        """
        channels = [(3, 8), (8, 16)]
        out_features = 10
        data = torch.autograd.Variable(torch.randn(16, 3, 32, 32))
        data_shape = (32, 32)

        special_genomes = [
            [
                [
                    [1],
                    [1, 1],
                    [1, 1, 1],
                    [1],
                    [2]
                ],
                [
                    [1],
                    [1, 1],
                    [1, 1],
                    [1, 1, 1],
                    [1, 1, 1, 1],
                    [1],
                    [2]
                ]
            ],
            [
                [
                    [0],
                    [0, 0],
                    [0, 0, 0],
                    [0],
                    [2]
                ],
                [
                    [0],
                    [0, 0],
                    [0, 0, 0],
                    [0, 0, 0, 0],
                    [0],
                    [2]
                ]
            ],
            [
                [
                    [0],
                    [0, 0],
                    [0, 0, 0],
                    [0],
                    [2]
                ],
                [
                    [1],
                    [1, 1],
                    [1, 1, 1],
                    [1, 1, 1, 1],
                    [1],
                    [2]
                ]
            ],
            [
                [
                    [1],
                    [1, 1],
                    [1, 1, 1],
                    [1],
                    [2]
                ],
                [
                    [0],
                    [0, 0],
                    [0, 0, 0],
                    [0, 0, 0, 0],
                    [0],
                    [2]
                ]
            ]
        ]

        for genome in special_genomes:
            net = EvoNetwork(genome, channels, out_features, data_shape, decoder='variable')
            net(data)
