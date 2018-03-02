# graph_similarity.py

import torch
import networkx as nx
from plugins.ged4py.algorithm import graph_edit_dist


def compute_k(eig):
    """
    Compute the value of k such that the sum of the first k eigenvalues divided by the sum of all eigenvalues is > 0.9.
    :param eig: torch.Tensor, 1D, the eigenvalues of some matrix.
    :return: int, k as described above.
    """
    total = torch.sum(eig)

    k_sum = 0
    for k in range(len(eig)):
        k_sum += eig[k]

        if (k_sum / total) > 0.9:
            return k


def degree_matrix(adj_m):
    """
    Caluclate D, the degree matrix given a dependancy graph.
    :param adj_m: torch.Tensor, 2D, (N x N), adjacency matrix.
    :return: torch.Tensor, 2D, (N x N), where N is the number of nodes in the graph.
    """
    D = torch.zeros((adj_m.size(0), adj_m.size(1)))

    row_sums = torch.sum(adj_m, dim=0)
    col_sums = torch.sum(adj_m, dim=1)

    for i, (row_sum, col_sum) in enumerate(zip(row_sums, col_sums)):
        D[i, i] = row_sum + col_sum

    return D


def am_to_lapacian(adj_m):
    """
    Computes the Laplacian based off a dependency graph.
    :param adj_m: torch.Tensor, 2D, (N x N), adjacency matrix.
    :return: torch.Tensor, 2D, (N x N) where N is the number of nodes in the graph.
    """
    return degree_matrix(adj_m) - adj_m


def dg_to_am(dg):
    """
    Turn the dependency graph into an adjacency matrix.
    :param dg: dict, dependency graph computed with ResidualGenomeDecoder.build_dependency_graph.
    :return: torch.Tensor, 2D, (N x N), where N is the number of nodes in the graph, describing the adjacencies.
    """
    nodes = max(dg.keys()) + 1
    am = torch.zeros((nodes, nodes))

    for sink, sources in dg.items():
        for source in sources:
            am[source, sink] = 1

    return am


def dg_to_nx(dg):
    """
    Turn the dependency graph into a networkx DiGraph.
    :param dg: dict, dependency graph computed with ResidualGenomeDecoder.build_dependency_graph.
    :return: networkx.DiGraph
    """
    g = nx.DiGraph()

    for sink, sources in dg.items():
        for source in sources:
            g.add_edge(source, sink)

    return g


def eigen_similarity_score(dg_1, dg_2):
    """
    Compute the similarity score using the eigenvalue decompositions of the two graphs' Laplacian matrices.

    Note: lower score means graphs are more similar.
    Note: this is only good for very large graphs (don't use).

    :param dg_1: dict, dependency graph computed with ResidualGenomeDecoder.build_dependency_graph.
    :param dg_2: dict, dependency graph computed with ResidualGenomeDecoder.build_dependency_graph.
    :return: float, [0, inf), 0 being a perfect match, higher values corresponding to more and more dissimilar graphs.
    """
    am_1, am_2 = dg_to_am(dg_1), dg_to_am(dg_2)  # Convert to adjacency matrices.
    L_1, L_2 = am_to_lapacian(am_1), am_to_lapacian(am_2)
    eig_1, eig_2 = torch.eig(L_1)[0][:, 0], torch.eig(L_2)[0][:, 0]
    k_1, k_2 = compute_k(eig_1), compute_k(eig_2)

    k = k_1 if k_1 < k_2 else k_2  # Choose the minimum of the two k values.

    # Return the sum of squares difference between the eigenvalues of the Laplacians.
    return sum([(eig_1[i] - eig_2[i]) * (eig_1[i] - eig_2[i]) for i in range(k)])


def ged_similarity_score(dg_1, dg_2):
    """
    Compute the similarity score using the graph edit distance (GED).

    Note: this will become very close for large graphs.
    Note: output normalized by the size of the graphs (shouldn't matter for us since they'll almost the same size).

    :param dg_1: dict, dependency graph computed with ResidualGenomeDecoder.build_dependency_graph.
    :param dg_2: dict, dependency graph computed with ResidualGenomeDecoder.build_dependency_graph.
    :return: float, [0, inf), 0 being a perfect match, higher values corresponding to more dissimilar graphs.
    """
    return graph_edit_dist.compare(dg_to_nx(dg_1), dg_to_nx(dg_2))


def demo():
    import time
    from evolution.residual_decoder import ResidualPhase
    genome1 = [[[1], [0, 0], [1, 1, 1], [0, 0, 0, 0], [1]]]
    genome2 = [[[0], [0, 0], [0, 1, 0], [0, 0, 0, 1], [1]]]
    genome3 = [[[0], [1, 0], [0, 0, 1], [0, 0, 0, 0], [1]]]  # Functionally the same as genome2.

    # Get dependency graphs from decoder static method.
    dg_1 = ResidualPhase.build_dependency_graph(genome1[0])  # ** Get the gene from the genome **
    dg_2 = ResidualPhase.build_dependency_graph(genome2[0])
    dg_3 = ResidualPhase.build_dependency_graph(genome3[0])

    print("Edit distance between 1 and 1: {}".format(ged_similarity_score(dg_1, dg_1)))
    print("Edit distance between 1 and 2: {}".format(ged_similarity_score(dg_1, dg_2)))
    print("Edit distance between 1 and 3: {}".format(ged_similarity_score(dg_1, dg_3)))
    print("Edit distance between 2 and 3: {}".format(ged_similarity_score(dg_2, dg_3)))

    start = time.time()
    for _ in range(200):
        ged_similarity_score(dg_1, dg_2)

    print("200 computations operations took {}s,".format(round(time.time() - start, 2)))


if __name__ == "__main__":
    demo()
