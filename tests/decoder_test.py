import torch
import unittest
from evolution import LOSComputationGraph, LOSHourGlassBlock, Identity


class TestLOSComputationGraph(unittest.TestCase):

    @staticmethod
    def get_residual(node, dependency):
        for d in dependency:
            if d.resolution == node.resolution and d.residual:
                return d

        return None

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

        # Assert that the second half (plus one) of the graph has incoming residuals, excluding the last node.
        for dependency in dependencies[len(dependencies) // 2 + 1:8]:
            self.assertEqual(2, len(dependency))
            self.assertTrue(dependency[0].residual or dependency[1].residual)

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

        # Make sure every node is only getting at most two connections.
        for i in range(len(dependencies)):
            if i <= 2:
                self.assertEqual(len(dependencies[i]), 1)

            elif 3 <= i <= 7:
                self.assertEqual(len(dependencies[i]), 2)

            elif i == 8:
                self.assertEqual(len(dependencies[i]), 1)

        # Make sure nodes that get residuals are getting them from the correct node.
        self.assertEqual(self.get_residual(nodes[8], dependencies[8]).idx, 4)
        self.assertEqual(self.get_residual(nodes[7], dependencies[7]).idx, 5)
        self.assertEqual(self.get_residual(nodes[6], dependencies[6]).idx, 2)
        self.assertEqual(self.get_residual(nodes[5], dependencies[5]).idx, 3)
        self.assertEqual(self.get_residual(nodes[4], dependencies[4]).idx, 0)
        self.assertEqual(self.get_residual(nodes[3], dependencies[3]).idx, 1)

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


if __name__ == "__main__":
    unittest.main()
