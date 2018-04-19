import torch
import random
import unittest
from collections import defaultdict
from evolution import LOSComputationGraph, LOSHourGlassBlock, LOSHourGlassDecoder, Identity


class TestLOSComputationGraph(unittest.TestCase):

    test_genome_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    test_genome_2 = [1, 2, 3, 2, 1, 2, 3, 2, 1]
    test_genome_3 = [1, 2, 3, 4, 4, 4]

    def test_make_graph(self):
        #
        # Original Hourglass structure.
        #
        graph = LOSComputationGraph.make_graph(self.test_genome_1)

        nodes, dependencies = zip(*graph.items())

        # Ensure we got an ordered dict.
        for i, node in enumerate(nodes):
            self.assertEqual(node.idx, i)

        # Nodes in the first half of the network should all have residual to be true.
        for node in nodes[:len(nodes) // 2]:
            self.assertTrue(node.residual)

        # Nodes should obey the formula for resolution 2 ^ (-(g - 1)) for gene value g.
        for node, gene in zip(nodes, self.test_genome_1):
            self.assertAlmostEqual(pow(2, -(gene - 1)), node.resolution, delta=1e-8)

        #
        # W graph.
        #
        graph = LOSComputationGraph.make_graph(self.test_genome_2)

        nodes, dependencies = zip(*graph.items())

        # First 6 nodes should all save a residual.
        for node in nodes[:6]:
            self.assertTrue(node.residual)

        # Make sure nodes that get residuals are getting them from the correct node.
        self.assertEqual(nodes[8].residual_node.idx, 4)
        self.assertEqual(nodes[7].residual_node.idx, 5)
        self.assertEqual(nodes[6].residual_node.idx, 2)
        self.assertEqual(nodes[5].residual_node.idx, 3)
        self.assertEqual(nodes[4].residual_node.idx, 0)
        self.assertEqual(nodes[3].residual_node.idx, 1)

        #
        # W graph, with no under connections.
        #
        graph = LOSComputationGraph.make_graph(self.test_genome_2, under_connect=False)

        nodes, dependencies = zip(*graph.items())

        for i in range(len(nodes)):
            if i <= 1 or 4 <= i <= 5:
                self.assertTrue(nodes[i].residual)

            else:
                self.assertFalse(nodes[i].residual)

        #
        # Uninteresting graph.
        #
        graph = LOSComputationGraph.make_graph(self.test_genome_3)

        nodes, dependencies = zip(*graph.items())

        for node in nodes:
            self.assertFalse(node.residual)


class TestLOSHourGlassBlock(unittest.TestCase):

    def test_constructor(self):
        test_data = torch.autograd.Variable(torch.randn(1, 64, 64, 64))

        #
        # Original Hourglass structure.
        #
        graph = LOSComputationGraph(TestLOSComputationGraph.test_genome_1)
        block = LOSHourGlassBlock(graph, 64, 128)

        # Make sure we can forward prop the data.
        output = block(test_data).data
        shape = output.size()

        self.assertEqual([1, 128, 64, 64], list(shape))

        samplers = list(block.samplers)

        self.assertTrue(isinstance(samplers.pop(0), Identity))
        self.assertTrue(isinstance(samplers.pop(), Identity))

        for i, sampler in enumerate(samplers):
            if i < len(samplers) // 2:
                self.assertEqual(2,  sampler.kernel_size)

            elif i >= len(samplers) // 2:
                self.assertEqual(2, sampler.scale_factor)

        #
        # W graph.
        #
        graph = LOSComputationGraph(TestLOSComputationGraph.test_genome_2)
        block = LOSHourGlassBlock(graph, 64, 128)

        output = block(test_data).data
        shape = output.size()

        self.assertEqual([1, 128, 64, 64], list(shape))

        samplers = list(block.samplers)

        self.assertTrue(isinstance(samplers.pop(0), Identity))
        self.assertTrue(isinstance(samplers.pop(), Identity))

        for i, sampler in enumerate(samplers):
            if 0 <= i <= 1 or 4 <= i <= 5:
                self.assertEqual(2, sampler.kernel_size)

            elif 2 <= i <= 3 or 6 <= i <= 7:
                self.assertEqual(2, sampler.scale_factor)

        #
        # Uninteresting graph.
        #
        graph = LOSComputationGraph(TestLOSComputationGraph.test_genome_3)
        block = LOSHourGlassBlock(graph, 64, 128)

        output = block(test_data)
        shape = output.size()

        self.assertTrue([1, 128, 64, 64], list(shape))

        samplers = block.samplers

        for i, sampler in enumerate(samplers):
            if i == 0 or 4 <= i <= 5:
                self.assertTrue(isinstance(sampler, Identity))

            elif 1 <= i <= 3:
                self.assertEqual(2, sampler.kernel_size)

            else:
                self.assertEqual(8, sampler.scale_factor)


class TestLOSHourGlassDecoder(unittest.TestCase):

    @staticmethod
    def get_random_genome(length=9, resolution_range=(1, 5)):
        """
        Generate a random hourglass genome.
        :param length: int, how long the genome should be.
        :param resolution_range: tuple, (int, int), inclusive range of resolutions.
            Formula for resolution i: 2^{-(i - 1)}
        :return: list, a genome.
        """
        def correct(i):
            if i < resolution_range[0]:
                return resolution_range[0]

            if i > resolution_range[1]:
                return resolution_range[1]

            return i

        genome = [random.randint(1, 2)]

        for j in range(1, length - 1):
            genome.append(correct(genome[j - 1] + random.randint(-2, 2)))

        return genome

    def test_random_genomes(self):
        random.seed(0)

        data = torch.autograd.Variable(torch.rand(1, 3, 256, 256))

        used = defaultdict(bool)

        for _ in range(500):
            genome = self.get_random_genome()

            while str(genome) in used:
                genome = self.get_random_genome()

            used[str(genome)] = True

            print("Evaluating {}".format(genome))

            model = LOSHourGlassDecoder(genome, 2, 2)
            out = model(data)
            map_sum = torch.sum(out[-1])
            map_sum.backward()


if __name__ == "__main__":
    unittest.main()
