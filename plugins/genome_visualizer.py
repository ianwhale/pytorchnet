# evo_visualizer.py
#

from graphviz import Digraph
from string import ascii_letters
from models.evonetwork import Phase


sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def subscript(number):
    """
    Convert a string number to subscript.
    :param number: string
    :return: string
    """
    return number.translate(sub)


def make_dot_genome(genome, rankdir="UD", format="pdf", title=None, filename="genome"):
    """
    Graphviz representation of network created by genome.
    :param genome: list of lists.
        *** Assumes genome is repaired. ***
    :return: graphviz dot object.
    """
    node_color = "lightblue"
    pool_color = "orange"
    phase_background_color = "gray"
    fc_color = "gray"

    structure = []

    # Build node ids and names to make building graph easier.
    for i, gene in enumerate(genome):
        prefix = "gene_" + str(i)
        phase = ("cluster_" + str(i + 1), "Phase " + str(i + 1))

        nodes = [(prefix + "_node_0", ascii_letters[i] + subscript("0"))] \
            + [(prefix + "_node_" + str(j + 1), ascii_letters[i] + subscript(str(j + 1))) for j in range(len(gene) + 1)]

        pool = (prefix + "_pool", "Pooling")

        residual = gene[-1][0] == 1

        all_zeros = True
        for dependency in gene[:-1]:
            if sum(dependency) > 0:
                all_zeros = False
                break

        edges = []
        if all_zeros:
            edges.append((nodes[0][0], nodes[-1][0]))  # Phase is skipped if everything is all zeros.

        else:
            graph = Phase.build_dependency_graph(gene)

            for sink, dependencies in graph.items():
                for source in dependencies:
                    edges.append((nodes[source][0], nodes[sink][0]))

        structure.append(
            {
                "nodes": nodes,
                "edges": edges,
                "pool": pool,
                "phase": phase,
                "all_zeros": all_zeros
            }
        )

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(format=format, filename=filename+'.gv', node_attr=node_attr, graph_attr=dict(size="12,12"))
    dot.attr(rankdir=rankdir)

    if title:
        dot.attr(label=title+"\n\n")
        dot.attr(labelloc='t')

    dot.node(str("input"), "Input")

    for j, struct in enumerate(structure):
        nodes = struct['nodes']
        edges = struct['edges']
        phase = struct['phase']
        pool = struct['pool']
        all_zeros = struct['all_zeros']

        # Add nodes.
        dot.node(nodes[0][0], nodes[0][1], fillcolor=node_color)

        if j > 0:
            dot.edge(structure[j - 1]['pool'][0], nodes[0][0])

        if not all_zeros:
            with dot.subgraph(name=phase[0]) as p:
                p.attr(fillcolor=phase_background_color, label=phase[1], fontcolor="black", style="filled")

                for i in range(1, len(nodes) - 1):
                    p.node(nodes[i][0], nodes[i][1], fillcolor=node_color)

        dot.node(nodes[-1][0], nodes[-1][1], fillcolor=node_color)
        dot.node(*pool, fillcolor=pool_color)

        # Add edges.
        for edge in edges:
            dot.edge(*edge)

        dot.edge(nodes[-1][0], pool[0])

    dot.edge("input", structure[0]['nodes'][0][0])

    dot.node("linear", "Linear", fillcolor=fc_color)
    dot.edge(structure[-1]['pool'][0], "linear")

    return dot


def demo():
    """
    Demonstrate visualizing a genome.
    """
    genome = [
        [
            [1],
            [0, 1],
            [1, 0, 1],
            [1]
        ],
        [
            [0],
            [0, 1],
            [0, 1, 0],
            [0, 0, 1, 0],
            [0]
        ],
        [
            [0],
            [0, 0],
            [0, 0, 0],
            [1]
        ]
    ]

    d = make_dot_genome(genome, title="Demo Genome", filename="test")
    d.view()


if __name__ == "__main__":
    demo()
