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

        # With this layout, there are 2^18 = 262144 possible genomes.
        return (
            (
                (r_bit(),),
                (r_bit(), r_bit()),
                (r_bit(), r_bit(), r_bit()),
                (r_bit(),)
            ),
            (
                (r_bit(),),
                (r_bit(), r_bit()),
                (r_bit(), r_bit(), r_bit()),
                (r_bit(), r_bit(), r_bit(), r_bit()),
                (r_bit(),)
            )
        )


class TestEvoNetwork(unittest.TestCase):
    def test_random_genomes(self):
        """
        Iterate through a bunch of genomes and make sure all of them run without error (best I could think of...).
        This will take around 3 minutes to run.
        """
        n = 5000  # This tests a tad less than a fifth of the possible genomes.

        channels = [(3, 8), (8, 16)]
        out_features = 10
        data = torch.autograd.Variable(torch.randn(16, 3, 32, 32))
        data_shape = (32, 32)
        g = GenomeGenerator()

        for _ in range(n):
            net = EvoNetwork(g.get_random_genome(), channels, out_features, data_shape)
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
                    [1]
                ],
                [
                    [1],
                    [1, 1],
                    [1, 1],
                    [1, 1, 1],
                    [1, 1, 1, 1],
                    [1]
                ]
            ],
            [
                [
                    [0],
                    [0, 0],
                    [0, 0, 0],
                    [0]
                ],
                [
                    [0],
                    [0, 0],
                    [0, 0, 0],
                    [0, 0, 0, 0],
                    [0]
                ]
            ],
            [
                [
                    [0],
                    [0, 0],
                    [0, 0, 0],
                    [0]
                ],
                [
                    [1],
                    [1, 1],
                    [1, 1, 1],
                    [1, 1, 1, 1],
                    [1]
                ]
            ],
            [
                [
                    [1],
                    [1, 1],
                    [1, 1, 1],
                    [1]
                ],
                [
                    [0],
                    [0, 0],
                    [0, 0, 0],
                    [0, 0, 0, 0],
                    [0]
                ]
            ]
        ]

        for genome in special_genomes:
            net = EvoNetwork(genome, channels, out_features, data_shape)
            net(data)
