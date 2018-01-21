import torch
import torch.nn as nn


class Node(nn.Module):
    """
    Basic computation unit.
    Does convolution, batchnorm, and relu.
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_size=3, padding=1, bias=False):
        """
        Constructor.
        Default arguments preserve dimensionality of input.

        :param in_channels: input to the node.
        :param out_channels: output channels from the node.
        :param stride: stride of convolution, default 1.
        :param kernel_size: size of convolution kernal, default 3.
        :param padding: amount of zero padding, default 1.
        :param bias: true to use bias, false to not.
        """
        super(Node, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Apply forward propagation operation.
        :param x: Variable, input.
        :return: Variable.
        """
        return self.model(x)


class Phase(nn.Module):
    """
    Represents a computation unit.
    Made up of Nodes.
    """
    def __init__(self, gene, in_channels, out_channels):
        """
        Constructor.
        :param gene: element of genome describing this phase.
                     It is assumed that the gene maps to a viable phase structure.
        """
        super(Phase, self).__init__()

        # This is used to make the input the correct number of channels.
        # TODO: Consider making this not require gradients and always have kernel = 1
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1x1.requires_gradients = False
        print(self.conv1x1.weight)
        print(self.conv1x1.bias)

        # Gene describes connections between nodes, so we need as many nodes as there are descriptors.
        self.nodes = [Node(out_channels, out_channels) for _ in range(len(gene) + 1)]

        self.dependency_graph = Phase.build_dependency_graph(gene)

        all_zeros = True
        for dependency in gene[:-1]:
            if sum(dependency) > 0:
                all_zeros = False
                break

        self.all_zeros = all_zeros
        self.residual = gene[-1][0] == 1
        self.nodes = nn.ModuleList(self.nodes)

    @staticmethod
    def build_dependency_graph(gene):
        """
        Build a graph describing the connections of a phase.
        "Repairs" made are as follows:
            - If a node has no input, but gives output, connect it to the input node (index 0 in outputs).
            - If a node has input, but no output, connect it to the output node (value returned from forward method).
        :param gene: gene describing the phase connections.
        :return: dict
        """
        graph = {}
        residual = gene[-1][0] == 1

        # First pass, build the graph without repairs.
        graph[1] = []
        for i in range(len(gene) - 1):
            graph[i + 2] = [j + 1 for j in range(len(gene[i])) if gene[i][j] == 1]

        graph[len(gene) + 1] = [0] if residual else []

        # Determine which nodes, if any, have no inputs and/or outputs.
        no_inputs = []
        no_outputs = []
        for i in range(1, len(gene) + 1):
            if len(graph[i]) == 0:
                no_inputs.append(i)

            has_output = False
            for j in range(i + 1, len(gene) + 2):
                if i in graph[j]:
                    has_output = True
                    break

            if not has_output:
                no_outputs.append(i)

        for node in no_outputs:
            if node not in no_inputs:
                # No outputs, but has inputs. Connect to output node.
                graph[len(gene) + 1].append(node)

        for node in no_inputs:
            if node not in no_outputs:
                # No inputs, but has outputs. Connect to input node.
                graph[node].append(0)

        return graph

    def forward(self, x):
        """
        Forward propagate through the nodes based on their dependency graph.
        :param x: Variable, input.
        :return: Variable.
        """
        if self.all_zeros:
            return self.conv1x1(x)

        outputs = [self.conv1x1(x)]

        for i in range(1, len(self.nodes)):
            if not self.dependency_graph[i]:  # Empty list.
                outputs.append(None)

            else:
                outputs.append(self.nodes[i](self.sum_dependencies(i, outputs)))

        return self.sum_dependencies(len(self.nodes), outputs)

    def sum_dependencies(self, i, outputs):
        """
        Sum over a node's dependencies.
        :param i: key of node's dependency list.
        :param outputs: outputs from previous i - 1 nodes.
        :return: Variable.
        """
        return sum((outputs[j] for j in self.dependency_graph[i]))


class EvoNetwork(nn.Module):
    """
    Entire network.
    Made up of Phases.
    """

    # TODO: Make this more general.

    def __init__(self, genome, channels, out_features, data_shape):
        """
        Network constructor.
        :param genome: list of genes, all assumes to be validated (repaired).
        :param channels: list of desired channel tuples.
        :param out_features: number of output features.
        """
        super(EvoNetwork, self).__init__()

        self.channels = channels
        self.data_shape = data_shape

        layers = []
        for gene, channel_tup in zip(genome, channels):
            layers.append(Phase(gene, channel_tup[0], channel_tup[1]))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Reduce dimension by half. TODO: Generalize this.

        self.model = nn.Sequential(*layers)

        # Do a test forward pass to determine the output shape of the evolved part of the network.
        shape = self.model(torch.autograd.Variable(torch.zeros(1, channels[0][0], *data_shape))).data.shape

        self.linear = nn.Linear(shape[1] * shape[2] * shape[3], out_features)

    def forward(self, x):
        """
        Forward propagation.
        :param x: Variable, input to network.
        :return: Variable.
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)

        return self.linear(x)


def demo():
    """
    Demo creating a single phase network.
    """
    # Genome should be a list of genes describing phase connection schemes.
    genome = [[
        [1],        # A_2 connections.
        [0, 0],     # A_3 connections.
        [1, 1, 1],  # A_4 connections.
        [1]         # A_5 connections (do we connect to A_0?)
    ]]

    # One input channel, 8 output channels.
    channels = [(3, 8)]

    out_features = 1
    data = torch.randn(3, 3, 32, 32)
    net = EvoNetwork(genome, channels, out_features, (32, 32))

    print(net(torch.autograd.Variable(data)))


if __name__ == "__main__":
    demo()
