# graph_similarity_test.py

import torch
import unittest
from plugins.graph_similarity import *


class TestGraphSimilarity(unittest.TestCase):
    dg_1 = {
        1: [0],
        2: [1],
        3: [0],
        4: [1, 2, 3],
        5: [],
        6: [0, 4]
    }

    dg_2 = {  # Same as dg_3.
        1: [],
        2: [0],
        3: [],
        4: [2],
        5: [4],
        6: [0, 5]
    }

    dg_3 = {  # Same as dg_2.
        1: [0],
        2: [],
        3: [1],
        4: [3],
        5: [],
        6: [0, 4]
    }

    def test_process_dg(self):
        am = dg_to_am(self.dg_1)

        expected = torch.Tensor([
            [0, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        self.assertTrue(torch.equal(expected, am))

        am = dg_to_am(self.dg_2)

        expected = torch.Tensor([
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        self.assertTrue(torch.equal(expected, am))

        am = dg_to_am(self.dg_3)

        expected = torch.Tensor([
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        self.assertTrue(torch.equal(expected, am))

    def test_degree_matrix(self):
        D = degree_matrix(dg_to_am(self.dg_1))

        expected = torch.diag(torch.Tensor([3, 3, 2, 2, 4, 0, 2]))

        self.assertTrue(torch.equal(expected, D))

        D = degree_matrix(dg_to_am(self.dg_2))

        expected = torch.diag(torch.Tensor([2, 0, 2, 0, 2, 2, 2]))

        self.assertTrue(torch.equal(expected, D))

        D = degree_matrix(dg_to_am(self.dg_3))

        expected = torch.diag(torch.Tensor([2, 2, 0, 2, 2, 0, 2]))

        self.assertTrue(torch.equal(expected, D))

    def test_compute_k(self):
        vec = torch.Tensor([i for i in range(10, 0, -1)])

        self.assertEqual(7, compute_k(vec))
