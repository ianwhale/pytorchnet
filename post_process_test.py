import os
import pickle
import matplotlib.pyplot as plt
from plugins.genome_visualizer import make_dot_genome


def make_network_title(individual, generation, nsga_details=True, show_genome=False):
    """
    Make the title based off properties of the individual.
    :param individual: nsgaII individual
    :param generation: int
    :param nsga_details: bool, true if we want rank and crowding distance in the title.
    :param show_genome: bool, true if we want to show the genome in the title.
    :return: str
    """
    accuracy = -individual.fitness[0]
    complexity = individual.fitness[1] if len(individual.fitness) > 1 else None

    title = "Individual Generation " + str(generation) \
        + " Address " + str(round(individual.address, 4)) \
        + "\nAccuracy: " + str(accuracy)

    if complexity:
        title += "\nComplexity: " + str(complexity)

    if nsga_details:
        title += "\nRank: " + str(individual.rank) \
            + "\nCrowding Distance: " + str(individual.crowding_dist)

    if show_genome:
        genome_str = ""
        for gene in individual.genome:
            genome_str += "-".join(["".join([str(int(i)) for i in node]) for node in gene]) + " "
        title += "\n" + " || ".join(genome_str.strip().split(" "))

    return title


def render_networks(population, generation, nsga_details=True, show_genome=False):
    """
    Renders the graphviz and image files of network architecture defined by a genome.
    :param population: list of nsga individuals.
    :param nsga_details: bool, true if we want rank and crowding distance in the title.
    :param show_genome: bool, true if we want the genome in the title.
    """
    base = "post"
    label = "gen_" + str(generation)

    output_dir = os.path.join(base, label, "networks")

    for individual in population:
        filename = label + "_addr_" + str(individual.address)
        path = os.path.join(output_dir, filename)

        title = make_network_title(individual, generation, nsga_details=nsga_details, show_genome=show_genome)

        viz = make_dot_genome(individual.genome, title=title)
        viz.render(path, view=False)


def make_plots(population, generation):
    """
    Renders plots of the objective(s) over generations.
    :param population: list of nsga individuals.
    """
    base = "post"
    label = "gen_" + str(generation)

    objectives = len(archive[0][0].fitness)

    if objectives != 1 and objectives != 2:
        raise ValueError("make_plots can only handle 1 or 2 objectives")

    output_dir = os.path.join(base, label, "plots")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if objectives == 1:
        # Only one objective, make a violin plot of fitnesses.
        path = os.path.join(output_dir, label + "_violin")
        raise NotImplementedError("Haven't done this yet.")

    else:
        # Two objectives, make a scatter plot.
        path = os.path.join(output_dir, label + "_objectives")

        non_dom_1 = []
        non_dom_2 = []

        dom_1 = []
        dom_2 = []

        for individual in population:
            if individual.rank == 1:
                non_dom_1.append(-individual.fitness[0])
                non_dom_2.append(individual.fitness[1])

            else:
                dom_1.append(-individual.fitness[0])
                dom_2.append(individual.fitness[1])

        fig, ax = plt.subplots()
        fig.suptitle("Generation {} Objectives".format(generation), fontsize=14, fontweight="bold")
        ax.plot(non_dom_1, non_dom_2, 'kd')
        ax.plot(dom_1, dom_2, 'ko')
        ax.plot()
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Complexity")
        plt.draw()

        try:
            fig.savefig(fname=path)

        except TypeError:
            fig.savefig(filename=path)


if __name__ == "__main__":
    archive = [pickle.load(open("gen0.pkl", "rb"))]

    for generation, population in enumerate(archive):
        render_networks(population, generation, show_genome=True)
        make_plots(population, generation)
