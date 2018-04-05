# decoder.py

import torch
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


class ChannelBasedDecoder(Decoder):
    """
    Channel based decoder that deals with encapsulating constructor logic.
    """
    def __init__(self, list_genome, channels):
        super().__init__(list_genome)

        self._model = None

        # First, we remove all inactive phases.
        self._genome = self.get_effective_genome(list_genome)
        self._channels = channels[:len(self._genome)]

        # If we had no active nodes, our model is just the identity, and we stop constructing.
        if not self._genome:
            self._model = Identity()

    @staticmethod
    def build_layers(phases):
        """
        Build up the layers with transitions.
        :param phases: list of phases
        :return: list of layers (the model).
        """
        layers = []
        last_phase = phases.pop()
        for phase in phases:
            layers.append(phase)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # TODO: Generalize this, or consider a new genome.

        layers.append(last_phase)
        return layers

    @staticmethod
    def get_effective_genome(genome):
        """
        Get only the parts of the genome that are active.
        :param genome: list, represents the genome
        :return: list
        """
        return [gene for gene in genome if phase_active(gene)]

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()


class ResidualGenomeDecoder(ChannelBasedDecoder):
    """
    Genetic CNN genome decoder with residual bit.
    """
    def __init__(self, list_genome, channels, preact=False):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        """
        super().__init__(list_genome, channels)

        if self._model is not None:
            return  # Exit if the parent constructor set the model.

        # Build up the appropriate number of phases.
        phases = []
        for idx, (gene, (in_channels, out_channels)) in enumerate(zip(self._genome, self._channels)):
            phases.append(ResidualPhase(gene, in_channels, out_channels, idx, preact=preact))

        self._model = nn.Sequential(*self.build_layers(phases))

    def get_model(self):
        """
        :return: nn.Module
        """
        return self._model


class ResidualPhase(nn.Module):
    """
    Residual Genome phase.
    """
    def __init__(self, gene, in_channels, out_channels, idx, preact=False):
        """
        Constructor.
        :param gene: list, element of genome describing connections in this phase.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :param idx: int, index in the network.
        :param preact: should we use the preactivation scheme?
        """
        super(ResidualPhase, self).__init__()

        self.channel_flag = in_channels != out_channels  # Flag to tell us if we need to increase channel size.
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1 if idx != 0 else 3, stride=1, bias=False)
        self.dependency_graph = ResidualPhase.build_dependency_graph(gene)

        if preact:
            node_constructor = PreactResidualNode

        else:
            node_constructor = ResidualNode

        nodes = []
        for i in range(len(gene)):
            if len(self.dependency_graph[i + 1]) > 0:
                nodes.append(node_constructor(out_channels, out_channels))
            else:
                nodes.append(None)  # Module list will ignore NoneType.

        self.nodes = nn.ModuleList(nodes)

        #
        # At this point, we know which nodes will be receiving input from where.
        # So, we build the 1x1 convolutions that will deal with the depth-wise concatenations.
        #

        conv1x1s = [Identity()] + [Identity() for _ in range(max(self.dependency_graph.keys()))]
        for node_idx, dependencies in self.dependency_graph.items():
            if len(dependencies) > 1:
                conv1x1s[node_idx] = \
                    nn.Conv2d(len(dependencies) * out_channels, out_channels, kernel_size=1, bias=False)

        self.processors = nn.ModuleList(conv1x1s)
        self.out = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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
        if self.channel_flag:
            x = self.first_conv(x)

        outputs = [x]

        for i in range(1, len(self.nodes) + 1):
            if not self.dependency_graph[i]:  # Empty list, no outputs to give.
                outputs.append(None)

            else:
                outputs.append(self.nodes[i - 1](self.process_dependencies(i, outputs)))

        return self.out(self.process_dependencies(len(self.nodes) + 1, outputs))

    def process_dependencies(self, node_idx, outputs):
        """
        Process dependencies with a depth-wise concatenation and
        :param node_idx: int,
        :param outputs: list, current outputs
        :return: Variable
        """
        return self.processors[node_idx](torch.cat([outputs[i] for i in self.dependency_graph[node_idx]], dim=1))


class ResidualNode(nn.Module):
    """
    Basic computation unit.
    Does convolution, batchnorm, and relu (in this order).
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_size=3, padding=1, bias=False):
        """
        Constructor.
        Default arguments preserve dimensionality of input.

        :param in_channels: input to the node.
        :param out_channels: output channels from the node.
        :param stride: stride of convolution, default 1.
        :param kernel_size: size of convolution kernel, default 3.
        :param padding: amount of zero padding, default 1.
        :param bias: true to use bias, false to not.
        """
        super(ResidualNode, self).__init__()

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


class PreactResidualNode(nn.Module):
    """
    Basic computation unit.
    Does batchnorm, relu, and convolution (in this order).
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_size=3, padding=1, bias=False):
        """
        Constructor.
        Default arguments preserve dimensionality of input.

        :param in_channels: input to the node.
        :param out_channels: output channels from the node.
        :param stride: stride of convolution, default 1.
        :param kernel_size: size of convolution kernel, default 3.
        :param padding: amount of zero padding, default 1.
        :param bias: true to use bias, false to not.
        """
        super(PreactResidualNode, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        )

    def forward(self, x):
        """
        Apply forward propagation operation.
        :param x: Variable, input.
        :return: Variable.
        """
        return self.model(x)


class VariableGenomeDecoder(ChannelBasedDecoder):
    """
    Residual decoding with extra integer for type of node inside the phase.
    This genome decoder produces networks that are a superset of ResidualGenomeDecoder networks.
    """
    RESIDUAL = 0
    PREACT_RESIDUAL = 1
    DENSE = 2

    def __init__(self, list_genome, channels):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network, and the type of phase.
        :param channels: list, list of tuples describing the channel size changes.
        """
        phase_types = [gene.pop() for gene in list_genome]

        super().__init__(list_genome, channels)

        if self._model is not None:
            return  # Exit if the parent constructor set the model.

        self._genome, self._types = self.get_effective_phases(list_genome, phase_types)
        self._channels = channels[:len(self._genome)]

        phases = []
        for idx, (gene, (in_channels, out_channels), phase_type) in enumerate(zip(self._genome,
                                                                                  self._channels,
                                                                                  self._types)):
            if phase_type == self.RESIDUAL:
                phases.append(ResidualPhase(gene, in_channels, out_channels, idx))

            elif phase_type == self.PREACT_RESIDUAL:
                phases.append(ResidualPhase(gene, in_channels, out_channels, idx, preact=True))

            elif phase_type == self.DENSE:
                phases.append(DensePhase(gene, in_channels, out_channels, idx))

            else:
                raise NotImplementedError("Phase type corresponding to {} not implemented.".format(phase_type))

        self._model = nn.Sequential(*self.build_layers(phases))

    @staticmethod
    def get_effective_phases(genome, phase_types):
        """
        Get only the phases that are active.
        Similar to ResidualDecoder.get_effective_genome but we need to consider phases too.
        :param genome: list, list of ints
        :param phase_types: list,
        :return:
        """
        effecive_genome = []
        effective_types = []

        for gene, phase_type in zip(genome, phase_types):
            if phase_active(gene):
                effecive_genome.append(gene)
                effective_types.append(*phase_type)

        return effecive_genome, effective_types

    def get_model(self):
        return self._model


class DenseGenomeDecoder(ChannelBasedDecoder):
    """
    Genetic CNN genome decoder with residual bit.
    """
    def __init__(self, list_genome, channels):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        """
        super().__init__(list_genome, channels)

        if self._model is not None:
            return  # Exit if the parent constructor set the model.

        # Build up the appropriate number of phases.
        phases = []
        for idx, (gene, (in_channels, out_channels)) in enumerate(zip(self._genome, self._channels)):
            phases.append(DensePhase(gene, in_channels, out_channels, idx))

        self._model = nn.Sequential(*self.build_layers(phases))

    @staticmethod
    def get_effective_genome(genome):
        """
        Get only the parts of the genome that are active.
        :param genome: list, represents the genome
        :return: list
        """
        return [gene for gene in genome if phase_active(gene)]

    def get_model(self):
        """
        :return: nn.Module
        """
        return self._model


class DensePhase(nn.Module):
    """
    Phase with nodes that operates like DenseNet's bottle necking and growth rate scheme.
    Refer to: https://arxiv.org/pdf/1608.06993.pdf
    """
    def __init__(self, gene, in_channels, out_channels, idx):
        """
        Constructor.
        :param gene: list, element of genome describing connections in this phase.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :param idx: int, index in the network.
        """
        super(DensePhase, self).__init__()

        self.in_channel_flag = in_channels != out_channels  # Flag to tell us if we need to increase channel size.
        self.out_channel_flag = out_channels != DenseNode.t
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1 if idx != 0 else 3, stride=1, bias=False)
        self.dependency_graph = ResidualPhase.build_dependency_graph(gene)

        channel_adjustment = 0

        for dep in self.dependency_graph[len(gene) + 1]:
            if dep == 0:
                channel_adjustment += out_channels

            else:
                channel_adjustment += DenseNode.t

        self.last_conv = nn.Conv2d(channel_adjustment, out_channels, kernel_size=1, stride=1, bias=False)

        nodes = []
        for i in range(len(gene)):
            if len(self.dependency_graph[i + 1]) > 0:
                channels = self.compute_channels(self.dependency_graph[i + 1], out_channels)
                nodes.append(DenseNode(channels))

            else:
                nodes.append(None)

        self.nodes = nn.ModuleList(nodes)
        self.out = nn.Sequential(
            self.last_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def compute_channels(dependency, out_channels):
        """
        Compute the number of channels incoming to a node.
        :param dependency: list, nodes that a particular node gets input from.
        :param out_channels: int, desired number of output channels from the phase.
        :return: int
        """
        channels = 0
        for d in dependency:
            if d == 0:
                channels += out_channels

            else:
                channels += DenseNode.t

        return channels

    def forward(self, x):
        if self.in_channel_flag:
            x = self.first_conv(x)

        outputs = [x]

        for i in range(1, len(self.nodes) + 1):
            if not self.dependency_graph[i]:  # Empty dependencies, no output to give.
                outputs.append(None)

            else:
                # Call the node on the depthwise concatenation of its inputs.
                outputs.append(self.nodes[i - 1](torch.cat([outputs[j] for j in self.dependency_graph[i]], dim=1)))

        if self.out_channel_flag and 0 in self.dependency_graph[len(self.nodes) + 1]:
            # Get the last nodes in the phase and change their channels to match the desired output.
            non_zero_dep = [dep for dep in self.dependency_graph[len(self.nodes) + 1] if dep != 0]

            return self.out(torch.cat([outputs[i] for i in non_zero_dep] + [outputs[0]], dim=1))

        if self.out_channel_flag:
            # Same as above, we just don't worry about the 0th node.
            return self.out(torch.cat([outputs[i] for i in self.dependency_graph[len(self.nodes) + 1]], dim=1))

        return self.out(torch.cat([outputs[i] for i in self.dependency_graph[len(self.nodes) + 1]]))


class DenseNode(nn.Module):
    """
    Node that operates like DenseNet layers.
    Refer to: https://arxiv.org/pdf/1608.06993.pdf
    """
    t = 32  # Growth rate fixed at 32 (a hyperparameter, although fixed in paper)
    k = 4   # Growth rate multiplier fixed at 4 (not a hyperparameter, this is from the definition of the dense layer).

    def __init__(self, in_channels):
        """
        Constructor.
        Only needs number of input channels, everything else is automatic from growth rate and DenseNet specs.
        :param in_channels: int, input channels.
        """
        super(DenseNode, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, self.t * self.k, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.t * self.k),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.t * self.k, self.t, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.model(x)


def phase_active(gene):
    """
    Determine if a phase is active.
    :param gene: list, gene describing a phase.
    :return: bool, true if active.
    """
    # The residual bit is not relevant in if a phase is active, so we ignore it, i.e. gene[:-1].
    return sum([sum(t) for t in gene[:-1]]) != 0


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


def demo():
    """
    Build a network and show its backprop graph..
    """
    from plugins.backprop_visualizer import make_dot_backprop
    from plugins.genome_visualizer import make_dot_genome
    genome = [
        [
            [1],
            [0, 0],
            [1, 0, 0],
            [1]
        ],
        [  # Phase will be ignored, there are no active connections (residual is not counted as active).
            [0],
            [0, 0],
            [0, 0, 0],
            [1]
        ],
        [
            [1],
            [0, 0],
            [0, 0, 0],
            [0, 0, 0, 0],
            [1]
        ]
    ]

    channels = [(3, 8), (8, 8), (8, 8)]
    data = torch.randn(16, 3, 32, 32)

    chopped = [gene[:-1] for gene in genome]
    make_dot_genome(chopped).view()

    model = DenseGenomeDecoder(genome, channels).get_model()
    out = model(torch.autograd.Variable(data))
    print(model)
    make_dot_backprop(out).view()


if __name__ == "__main__":
    demo()
