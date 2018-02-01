# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 14:40:18 2017

@author: Dhebar
@modified by: Zhichao Lu
modified for for structure search for the deep neural network
the selection is based on the 'dominance' check. Rank information is 
considered obsolete here
"""

#....NSGA-II main code....

import nsga2_classes
import numpy as np
import random
import parameter_inputs
import nsga_funcs as nf
import global_vars
import timeit
import pickle
import os


if __name__ == "__main__":

    random.seed(9001)
    global_vars.declare_global_vars()
    global_vars.params = parameter_inputs.input_parameters()
    cons = parameter_inputs.input_constants()
    print('pop_size = %d' % global_vars.params.pop_size)
    start_time = timeit.default_timer()
    doResume = False
    for run in range(global_vars.params.max_runs):
        print('run = %d'%(run))
        print('initialize')
        archive = [] # stores every exactly evaluated solutions
        print('gen = %d'%0)
        if doResume:
            resume_gen = 18 # resume from which generation
            parent_pop = []
            archive = []
            with open("output_save/run0/pop_gen{}.pkl".format(resume_gen),"rb") as f:
                parent_pop = pickle.load(f)

            with open("output_save/run0/archive_gen{}.pkl".format(resume_gen),"rb") as f:
                archive = pickle.load(f)

            for i in range(resume_gen+1,global_vars.params.max_gen):
                print('gen = %d' % i)
                child_pop = nf.selection_ednn(parent_pop)
                nf.mutation_pop_bin(child_pop)
                nf.fill_genome(child_pop)
                child_pop, archive = nf.compute_fitness_pop_ednn(child_pop, archive)
                mixed_pop = parent_pop + child_pop
                parent_pop = nf.fill_nondominated_sort(mixed_pop)
                nf.statistics(parent_pop, archive, run, i)
        else:
            parent_pop = nf.initialize_pop_ednn(global_vars.params.pop_size)
            nf.fill_genome(parent_pop)
            parent_pop,archive  = nf.compute_fitness_pop_ednn(parent_pop,archive)
            nf.assign_rank_crowding_distance(parent_pop)
            nf.statistics(parent_pop,archive,run,0)
            # for j in range(global_vars.params.pop_size):
            #     print('Indv = %d, accuracy = %f, complexity = %d' % (j, parent_pop[j].fitness[0], parent_pop[j].fitness[1]))
            # putting every exactly evaluated individual into archive
            # generation starts
            for i in range(1,global_vars.params.max_gen):
                print('gen = %d'%i)
                child_pop = nf.selection_ednn(parent_pop)
                #child_pop = nf.randomSelection(parent_pop)
                #child_pop = nf.tournamentSelection2(parent_pop)
                #child_pop = nf.selection(parent_pop)
                #nf.mutation_pop(child_pop)
                nf.mutation_pop_bin(child_pop)
                nf.fill_genome(child_pop)
                child_pop,archive = nf.compute_fitness_pop_ednn(child_pop,archive)
                mixed_pop = parent_pop + child_pop
                parent_pop = nf.fill_nondominated_sort(mixed_pop)
                nf.statistics(parent_pop,archive,run,i)
            # for plotting the objectives 
    #         for j in range(global_vars.params.pop_size):
    #             print('Indv = %d, accuracy = %f, complexity = %d'%(j,parent_pop[j].fitness[0],parent_pop[j].fitness[1]))
    # nf.write_final_pop_obj(parent_pop,run+1)
        
    end_time = timeit.default_timer()
    print('Execution time = %f hours' %((end_time - start_time)/3600))


