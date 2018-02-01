# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 15:18:39 2017

@author: Dhebar
@modified: zhichao
"""

#....nsga2 functions....
import nsga2_classes
import parameter_inputs
import random
import numpy as np
import sys
import math
import copy
import os
import pickle
import global_vars
import subprocess
import time
import shutil
import test_problems

#global_vars.params = parameter_inputs.input_parameters()
cons = parameter_inputs.input_constants()

def initialize_pop(pop_size):
    #...initializes the population pool
    pop = []
    for i in range(pop_size):
        pop.append(nsga2_classes.Individual())
        pop[i].xreal = np.array([0.0]*global_vars.params.n_var)
        pop[i].fitness = np.array([0.0]*global_vars.params.n_obj)
        pop[i].address = i
        for j in range(global_vars.params.n_var):
            pop[i].xreal[j] = random.uniform(global_vars.params.Lbound[j],global_vars.params.Ubound[j])
    return pop
    
def initialize_pop_ednn(pop_size):
    #...initializes the population pool
    pop = []
    # the following two lines calculate the total number of variables,which depends on the coding scheme
    n_var_stage = (global_vars.params.num_operator-1)*(global_vars.params.num_operator)/2 + 1
    n_var = int(n_var_stage*global_vars.params.num_stages)

    for i in range(pop_size):
        pop.append(nsga2_classes.Individual())
        pop[i].xreal = np.array([0.0]*n_var)
        pop[i].fitness = np.array([0.0]*global_vars.params.n_obj)
        pop[i].address = i
        for j in range(n_var):
            #pop[i].xreal[j] = random.uniform(global_vars.params.Lbound[j],global_vars.params.Ubound[j])
            if (random.random() < 0.5):
                pop[i].xreal[j] = 0
            else:
                pop[i].xreal[j] = 1
        pop[i].pt = 1
    return pop
        
#def compute_fitness_pop(pop):
#    for i in range(len(pop)):
#        test_problems.compute_fitness_ind(pop[i])
#    return

def compute_fitness_pop(pop):

    for i in range(len(pop)):
        pop[i].fitness[0] = np.random.randint(5)
        pop[i].fitness[1] = np.sum(pop[i].xreal)
    return

def compute_fitness_pop_ednn(pop,archive):
    pop1 = [] # this is the population that has a copy in the archive
    pop2 = [] # this is the population that actually needs to be evaluated in parallel
    for i in range(len(pop)):
        # check to see if this solution has been evaluated previously
        isPrevEvaled = False
        for member in archive:
            if pop[i].genome == member.genome:
                pop[i].fitness = member.fitness
                isPrevEvaled = True
                pop[i].pt = 1
                pop1.append(pop[i])
                break
        if (not isPrevEvaled):
            # append the indv for parallel evaluation
            pop2.append(pop[i])
    # if the pop2 is not empty
    if len(pop2) > 0:
        # 1st delete the input_file directory w/ all contents
        if os.path.exists("input_file"):
            shutil.rmtree("input_file")
            os.makedirs("input_file")
        else:
            os.makedirs("input_file")
        # 2nd delete the output_file directory w/ all contents
        if os.path.exists("output_file"):
            shutil.rmtree("output_file")
            os.makedirs("output_file")
        else:
            os.makedirs("output_file")
        # prepare the input file for parallel evaluation
        for j in range(len(pop2)):
            with open("input_file/input%d.pkl" % j, "wb") as f:
                pickle.dump(pop2[j].genome, f)
        # run parallel evaluation in hpcc - qsub bash commend
        subprocess.call("qsub evaluate_pytorchnet.qsub",shell=True)
        # wait for the parallel execution to finish
        start = time.time()
        print("waiting for the HPCC parallel evaluation to start...")
        time.sleep(300)
        while True:
            time.sleep(60)  # pause python for 60 seconds before start to check output files from hpcc
            num_completed = 0
            for k in range(len(pop2)):
                if os.path.isfile("output_file/output%d.pkl" % k):
                    num_completed += 1
            print("[{0}/{1}] completed by HPCC".format(num_completed,len(pop2)))
            end = time.time()
            # exit the loop if all solutions in pop2 evaluated or
            # the maximum time (2 hours) is reached
            if (num_completed >= len(pop2)) or ((end - start)/3600 > 12):
                break
        print("HPCC parallel execution time = {} mins".format((end - start)/60))
        # hpcc finished evaluation, gather output information
        for j in range(len(pop2)):
            # extra check for walltime limit results
            if os.path.exists("output_file/output%d.pkl" % j):
                with open("output_file/output%d.pkl" % j, "rb") as f:
                    pop2[j].fitness[0] = pickle.load(f)
                pop2[j].fitness[0] = -pop2[j].fitness[0]
            else:
                pop2[j].fitness[0] = 0
            pop2[j].fitness[1] = calculate_complexity_ind(pop2[j])
        # merge the pop2 with archive
        new_archive = archive + pop2
    else:
        print("All offspring has been previous evaluated")
        new_archive = archive
    # merge the pop1 and pop2
    new_pop = pop1 + pop2
    return new_pop,new_archive

def dominates(p,q):
    #...check if p dominates q
    #..returns 1 of p dominates q, -1 if q dominates p, 0 if both are non-dominated
    flag1 = 0
    flag2 = 0
    for i in range(global_vars.params.n_obj):
        if p.fitness[i] < q.fitness[i]:
            flag1 = 1
        else:
            if p.fitness[i] > q.fitness[i]:
                flag2 = 1
        
    if (flag1 == 1) and (flag2 == 0):
        return 1
    else:
        if (flag1 == 0) and (flag2 == 1):
            return -1
        else:
            return 0


def sort_wrt_obj(f,j):
    obj_values = []
    for p in f:
        obj_values.append(p.fitness[j])
        
    r = range(len(f))
    g = [x for _,x in sorted(zip(obj_values,r))]
    
    return g,max(obj_values),min(obj_values)
           
def crowding_distance(F):
    #....
    for f in F:
        f_len = len(f)
        if (f_len <= 2):
            for p in f:
                p.crowding_dist = float('inf')
        else:
            for p in f:
                p.crowding_dist = 0.0
    
            for j in range(global_vars.params.n_obj):
                f_id, max_j, min_j = sort_wrt_obj(f,j)
                f[f_id[0]].crowding_dist = float("inf")
                #f[f_id[-1]].crowding_dist = float("inf")
                if max_j != min_j:
                    for i in range(1,f_len-1):
                        f[f_id[i]].crowding_dist = f[f_id[i]].crowding_dist + \
                        (f[f_id[i+1]].fitness[j] - f[f_id[i-1]].fitness[j])/(max_j - min_j)
    
    return




def assign_rank_crowding_distance(pop):
    F = []#...list of fronts...
    f = []

    for p in pop:
        p.S_dom = []
        p.n_dom = 0
        for q in pop:
            dom_p_q = dominates(p,q)
            if (dom_p_q == 1):
                p.S_dom.append(q)
            elif (dom_p_q == -1):
                p.n_dom = p.n_dom + 1
                
        if p.n_dom == 0:
            p.rank = 1
            f.append(p)
        
    F.append(f)
    i = 0
    while (len(F[i]) != 0):
        Q = []
        for p in F[i]:
            for q in p.S_dom:
                q.n_dom = q.n_dom - 1
                if q.n_dom == 0:
                    q.rank = i + 2
                    Q.append(q)
        i = i+1
        F.append(Q)
        
    if len(F[-1]) == 0:
        del(F[-1])
        

    crowding_distance(F)
      
    return F

def sort_wrt_crowding_dist(f):
    c_dist_vals = []
    for p in f:
        c_dist_vals.append(p.crowding_dist)
        
    r = range(len(f))
    f_new_id = [x for _,x in sorted(zip(c_dist_vals,r),reverse=True)]
        
    return f_new_id


def tour_select(ind1, ind2):
    #print 'doing binary tournament selection'
    dom_1_2 = dominates(ind1,ind2)
    if (dom_1_2 == 1):
        return(ind1)
    elif (dom_1_2 == -1):
        return(ind2)
    else:
        if (ind1.crowding_dist > ind2.crowding_dist):
            return(ind1)
        elif (ind2.crowding_dist > ind1.crowding_dist):
            return(ind2)
        elif (random.random() > 0.5):
            return(ind1)
        else:
            return(ind2)


# modified routine from original nsga2 selection
def selection_ednn(old_pop):
    pop_size = len(old_pop)
    a1 = np.random.permutation(range(pop_size))
    a2 = np.random.permutation(range(pop_size))
    new_pop = [0] * pop_size
    for i in range(0, pop_size, 4):
        parent1 = tour_select(old_pop[a1[i]], old_pop[a1[i + 1]])
        parent2 = tour_select(old_pop[a1[i + 2]], old_pop[a1[i + 3]])
        # create first kid using within stage crossover
        new_pop[i] = crossover_within_stage(parent1,parent2)
        if global_vars.params.num_stages > 1:
            new_pop[i + 1] = crossover_between_stage(parent1, parent2)
        else:
            new_pop[i + 1] = crossover_within_stage(parent1, parent2)
        # create the second kid using between stage crossover
        # new_pop[i + 1] = crossover_between_stage(parent1,parent2)
        # new_pop[i + 1] = crossover_within_stage(parent1, parent2)
        #new_pop[i], new_pop[i + 1] = crossover(parent1, parent2)

        parent1 = tour_select(old_pop[a2[i]], old_pop[a2[i + 1]])
        parent2 = tour_select(old_pop[a2[i + 2]], old_pop[a2[i + 3]])
        # create first kid using within stage crossover
        new_pop[i + 2] = crossover_within_stage(parent1, parent2)
        if global_vars.params.num_stages > 1:
            new_pop[i + 3] = crossover_between_stage(parent1, parent2)
        else:
            new_pop[i + 3] = crossover_within_stage(parent1, parent2)

        # create the second kid using between stage crossover
        # new_pop[i + 3] = crossover_between_stage(parent1, parent2)
        # new_pop[i + 3] = crossover_within_stage(parent1, parent2)
        #new_pop[i + 2], new_pop[i + 3] = crossover(parent1, parent2)

    # ..address assignment...
    for i in range(len(new_pop)):
        new_pop[i].address = i

    return new_pop

def crossover_between_stage(parent1,parent2):
    n_var_stage = int((global_vars.params.num_operator - 1) * (global_vars.params.num_operator) / 2 + 1)
    n_var = int(n_var_stage * global_vars.params.num_stages)
    child = nsga2_classes.Individual()
    child.xreal = np.array([0.0] * n_var)
    child.fitness = np.array([0.0] * global_vars.params.n_obj)
    # loop through the stages
    for stage in range(global_vars.params.num_stages):
        if (random.random() < 0.5):
            child.xreal[stage*n_var_stage:stage*n_var_stage+n_var_stage] = parent1.xreal[stage*n_var_stage:stage*n_var_stage+n_var_stage]
        else:
            child.xreal[stage*n_var_stage:stage*n_var_stage+n_var_stage] = parent2.xreal[stage*n_var_stage:stage*n_var_stage+n_var_stage]
    # print('parent1 = ')
    # print(parent1.xreal)
    # print('parent2 = ')
    # print(parent2.xreal)
    # print('child = ')
    # print(child.xreal)
    # record parents' information
    child.pt = 0
    child.parent1_genome = parent1.genome
    child.parent2_genome = parent2.genome
    child.parent1_fitness = parent1.fitness
    child.parent2_fitness = parent2.fitness
    return child

def crossover_within_stage(parent1,parent2):
    # customized binary crossover
    # requires two parents, produce one offspring
    # 1st principle:
    # preserve the common bits between the two parents
    # 2nd principle:
    # maintain relative the same complexity

    n_var_stage = int((global_vars.params.num_operator - 1) * (global_vars.params.num_operator) / 2 + 1)
    n_var = int(n_var_stage * global_vars.params.num_stages)
    child = nsga2_classes.Individual()
    child.xreal = np.array([0.0] * n_var)
    child.fitness = np.array([0.0] * global_vars.params.n_obj)
    # loop through the number of vars
    common_bits = []
    for bit in range(n_var):
        # 1st principle
        if (parent1.xreal[bit] == parent2.xreal[bit]):
            child.xreal[bit] = parent1.xreal[bit]
            common_bits.append(bit)
        else:
            if (random.random() < 0.5):
                child.xreal[bit] = parent1.xreal[bit]
            else:
                child.xreal[bit] = parent2.xreal[bit]
    # 2nd principle
    num_ones_parent1 = np.sum(parent1.xreal)
    num_ones_parent2 = np.sum(parent2.xreal)
    num_ones_child = np.sum(child.xreal)

    # convert common_bit from list to numpy array
    # depending on if there are common bits between the two parents
    # if there is common bits, we need to make sure common bits are not changed
    if len(common_bits) > 0:
        common_bits = np.asarray(common_bits)
        # in case when number of ones in child is less the minimum of the two parents
        if num_ones_child < np.minimum(num_ones_parent1,num_ones_parent2):
            zeros_idx = np.where(child.xreal == 0)[0] # extract the bit with 0 value
            common_zero_bits = np.where(parent1.xreal[common_bits] == 0)[0] # extract the bits with 0 value in common bits
            common_zero_bits = common_bits[common_zero_bits]
            idx_del = []
            for common_zero in common_zero_bits:
                idx_del.append(np.where(zeros_idx == common_zero)[0])
            zeros_idx = np.delete(zeros_idx,idx_del)
            #zeros_idx = np.unique(np.append(zeros_idx,common_zero_bits)) # check if they are shared in common between parents
            np.random.shuffle(zeros_idx) # randomly shuffle the bit idx with 0 value
            zeros_idx = zeros_idx[0:int(np.minimum(num_ones_parent1,num_ones_parent2) - num_ones_child)]
            for zero_bit in zeros_idx:
                child.xreal[zero_bit] = 1.0
        # in case when number of ones in child is greater than the maximum of the two parents
        elif num_ones_child > np.maximum(num_ones_parent1,num_ones_parent2):
            ones_idx = np.where(child.xreal == 1.0)[0] # extract the bit with 1 value
            common_one_bits = np.where(parent1.xreal[common_bits] == 1.0)[0]  # extract the bits with 1 value in common bits
            common_one_bits = common_bits[common_one_bits]
            idx_del = []
            for common_one in common_one_bits:
                idx_del.append(np.where(ones_idx == common_one)[0])
            ones_idx = np.delete(ones_idx,np.asarray(idx_del))
            #ones_idx = np.unique(np.append(ones_idx,common_one_bits))  # check if they are shared in common between parents
            np.random.shuffle(ones_idx)  # randomly shuffle the bit idx with 1 value
            ones_idx = ones_idx[0:int(num_ones_child - np.maximum(num_ones_parent1, num_ones_parent2))]
            for one_bit in ones_idx:
                child.xreal[one_bit] = 0.0
    # if there is no common bits, we can change any bits to fulfill the 2nd principle
    else:
        # in case when number of ones in child is less the minimum of the two parents
        if num_ones_child < np.minimum(num_ones_parent1, num_ones_parent2):
            zeros_idx = np.where(child.xreal == 0)[0]  # extract the bit with 0 value
            np.random.shuffle(zeros_idx)  # randomly shuffle the bit idx with 0 value
            zeros_idx = zeros_idx[0:int(np.minimum(num_ones_parent1, num_ones_parent2) - num_ones_child)]
            for zero_bit in zeros_idx:
                child.xreal[zero_bit] = 1.0
        # in case when number of ones in child is greater than the maximum of the two parents
        elif num_ones_child > np.maximum(num_ones_parent1, num_ones_parent2):
            ones_idx = np.where(child.xreal == 1.0)[0]  # extract the bit with 1 value
            np.random.shuffle(ones_idx)  # randomly shuffle the bit idx with 1 value
            ones_idx = ones_idx[0:int(num_ones_child - np.maximum(num_ones_parent1, num_ones_parent2))]
            for one_bit in ones_idx:
                child.xreal[one_bit] = 0.0
    
    # print('parent1 = ')
    # print(parent1.xreal)
    # print('parent2 = ')
    # print(parent2.xreal)
    # print('child = ')
    # print(child.xreal)
        # record parents' information
    child.pt = 0
    child.parent1_genome = parent1.genome
    child.parent2_genome = parent2.genome
    child.parent1_fitness = parent1.fitness
    child.parent2_fitness = parent2.fitness
    return child

def crossover(parent1,parent2):
    child1 = nsga2_classes.Individual()
    child2 = nsga2_classes.Individual()
    child1.xreal = np.array([0.0]*global_vars.params.n_var)
    child2.xreal = np.array([0.0]*global_vars.params.n_var)
    child1.fitness = np.array([0.0]*global_vars.params.n_obj)
    child2.fitness = np.array([0.0]*global_vars.params.n_obj)
    
    if (random.random() < global_vars.params.p_xover):
        for i in range(global_vars.params.n_var):
            if (random.random() < 0.5):
                if (abs(parent1.xreal[i] - parent2.xreal[i]) > cons.EPS):
                    if (parent1.xreal[i] < parent2.xreal[i]):
                        y1 = parent1.xreal[i]
                        y2 = parent2.xreal[i]
                    else:
                        y1 = parent2.xreal[i]
                        y2 = parent1.xreal[i]
                    
                    y_L = global_vars.params.Lbound[i]
                    y_U = global_vars.params.Ubound[i]
                    rand = random.random()
                    beta = 1.0 + (2.0*(y1 - y_L)/(y2 - y_L))
                    alpha = 2.0 - math.pow(beta,-(global_vars.params.eta_xover + 1))
                    if (rand <= (1.0/alpha)):
                        beta_q = math.pow((rand*alpha),
                                          (1.0/(global_vars.params.eta_xover+1.0)))
                    else:
                        beta_q = math.pow((1.0/(2.0 - rand*alpha)),
                                    (1.0/(global_vars.params.eta_xover+1.0)))
                
                    c1 = 0.5*((y1+y2) - beta_q*(y2-y1))
                    beta = 1.0 + (2.0*(y_U-y2)/(y2-y1))
                    alpha = 2.0 - math.pow(beta,-(global_vars.params.eta_xover+1.0))
                    if rand <= (1.0/alpha):
                        beta_q = math.pow((rand*alpha),
                                          (1.0/(global_vars.params.eta_xover+1.0)))
                    else:
                        beta_q = math.pow((1.0/(2.0-rand*alpha)),
                                          (1.0/(global_vars.params.eta_xover+1.0)))
                    c2 = 0.5*((y1+y2)+beta_q*(y2-y1))
                    if (c1 < y_L):
                        c1 = y_L
                    if (c2 < y_L):
                        c2 = y_L
                    if (c1 > y_U):
                        c1 = y_U
                    if (c2 > y_U):
                        c2 = y_U
                        
                    if (random.random() <= 0.5):
                        child1.xreal[i] = c2
                        child2.xreal[i] = c1
                    else:
                        child1.xreal[i] = c1
                        child2.xreal[i] = c2
                        
                else:
                    child1.xreal[i] = parent1.xreal[i]
                    child2.xreal[i] = parent2.xreal[i]
            else:
                child1.xreal[i] = parent1.xreal[i]
                child2.xreal[i] = parent2.xreal[i]
    else:
        child1.xreal = np.array(list(parent1.xreal))
        child2.xreal = np.array(list(parent2.xreal))
        
    return child1,child2

def planar_PCX(parent1,parent2,parent3):
    # planar parent centric crossover
    # This code is written such that the parent1 is chosen as the index parent
    # three parents are needed for crossover
    sigma1, sigma2 = 0, 0.5
    child1 = nsga2_classes.Individual()
    child1.xreal = np.array([0.0]*global_vars.params.n_var)
    child1.fitness = np.array([0.0]*global_vars.params.n_obj)
    # the centroid of the parents
    temp1 = np.array(list(parent1.xreal))
    temp2 = np.array(list(parent2.xreal))
    temp3 = np.array(list(parent3.xreal))
    g = np.mean(np.array([temp1,temp2,temp3]),axis=0)
    
    if (random.random() < global_vars.params.p_xover):
        child1.xreal = (temp1 + sigma1*np.random.randn()*(temp1 - g) +
                        sigma2*np.random.randn()*(temp2 - temp3))
    else:
        child1.xreal = temp1
    
    for i in range(global_vars.params.n_var):
        if child1.xreal[i] < global_vars.params.Lbound[i]:
            child1.xreal[i] = global_vars.params.Lbound[i]
        if child1.xreal[i] > global_vars.params.Ubound[i]:
            child1.xreal[i] = global_vars.params.Ubound[i]
            
    return child1

#Routine for tournament selection, it creates a new_pop from old_pop by
#    performing tournament selection and the crossover  
def selection(old_pop):
    pop_size = len(old_pop)
    a1 = np.random.permutation(range(pop_size))
    a2 = np.random.permutation(range(pop_size))
    new_pop = [0]*pop_size
    for i in range(0,pop_size,4):
        parent1 = tour_select(old_pop[a1[i]],old_pop[a1[i+1]])
        parent2 = tour_select(old_pop[a1[i+2]],old_pop[a1[i+3]])
        new_pop[i], new_pop[i+1] = crossover(parent1,parent2)
        
        parent1 = tour_select(old_pop[a2[i]],old_pop[a2[i+1]])
        parent2 = tour_select(old_pop[a2[i+2]],old_pop[a2[i+3]])
        new_pop[i+2], new_pop[i+3] = crossover(parent1,parent2)
    
    
    #..address assignment...
    for i in range(len(new_pop)):
        new_pop[i].address = i
        
    return new_pop

#Routine for tournament selection, it creates a new_pop from old_pop by
#    performing tournament selection and the Vectorized crossover (DE or PCX)
def tournamentSelection2(old_pop):
    pop_size = len(old_pop)
    a1 = np.random.permutation(range(pop_size))
    a2 = np.random.permutation(range(pop_size))
    a3 = np.random.permutation(range(pop_size))
    new_pop = [0]*pop_size
    for i in range(0,pop_size,6):
        parent1 = tour_select(old_pop[a1[i]],old_pop[a1[i+1]])
        parent2 = tour_select(old_pop[a1[i+2]],old_pop[a1[i+3]])
        parent3 = tour_select(old_pop[a1[i+4]],old_pop[a1[i+5]])
        new_pop[i] = planar_PCX(parent1,parent2,parent3)
        new_pop[i+1] = planar_PCX(parent1,parent2,parent3)

        parent1 = tour_select(old_pop[a2[i]],old_pop[a2[i+1]])
        parent2 = tour_select(old_pop[a2[i+2]],old_pop[a2[i+3]])
        parent3 = tour_select(old_pop[a2[i+4]],old_pop[a2[i+5]])
        new_pop[i+2] = planar_PCX(parent1,parent2,parent3)
        new_pop[i+3] = planar_PCX(parent1,parent2,parent3)

        parent1 = tour_select(old_pop[a3[i]],old_pop[a3[i+1]])
        parent2 = tour_select(old_pop[a3[i+2]],old_pop[a3[i+3]])
        parent3 = tour_select(old_pop[a3[i+4]],old_pop[a3[i+5]])
        new_pop[i+4] = planar_PCX(parent1,parent2,parent3)
        new_pop[i+5] = planar_PCX(parent1,parent2,parent3)

    #..address assignment...
    for i in range(len(new_pop)):
        new_pop[i].address = i

    return new_pop

#Routine for tournament selection, it creates a new_pop from old_pop by
#    performing tournament selection and the Vectorized crossover (DE or PCX)
def randomSelection(old_pop):
    pop_size = len(old_pop)
    a1 = np.random.permutation(range(pop_size))
    a2 = np.random.permutation(range(pop_size))
    a3 = np.random.permutation(range(pop_size))
    new_pop = [0]*pop_size
    for i in range(pop_size):
        parent1 = old_pop[a1[i]]
        parent2 = old_pop[a2[i]]
        parent3 = old_pop[a3[i]]
        new_pop[i] = planar_PCX(parent1,parent2,parent3)

    #..address assignment...
    for i in range(len(new_pop)):
        new_pop[i].address = i

    return new_pop

def mutation_ind(ind):
    for j in range(global_vars.params.n_var):
        if (random.random() < global_vars.params.p_mut):
            y = ind.xreal[j]
            y_L = global_vars.params.Lbound[j]
            y_U = global_vars.params.Ubound[j]
            delta1 = (y-y_L)/(y_U - y_L)
            delta2 = (y_U - y)/(y_U - y_L)
            rnd = random.random()
            mut_pow = 1.0/(global_vars.params.eta_mut + 1.0)
            if (rnd <= 0.5):
                xy = 1.0 - delta1
                val = 2.0*rnd + (1.0 - 2.0*rnd)*(
                math.pow(xy,(global_vars.params.eta_mut + 1.0)))
                delta_q = math.pow(val,mut_pow) - 1.0
            else:
                xy = 1.0-delta2
                val = 2.0*(1.0-rnd)+2.0*(rnd-0.5)*(
                math.pow(xy,(global_vars.params.eta_mut+1.0)))
                delta_q = 1.0 - (math.pow(val,mut_pow))
                
            y = y + delta_q*(y_U - y_L)
            if (y < y_L):
                y = y_L
            if (y > y_U):
                y = y_U
                
            ind.xreal[j] = y
    return
        
def mutation_pop(pop):
    for p in pop:
        mutation_ind(p)
        
    return

def mutation_ind_bin(ind):
    n_var = len(ind.xreal)
    for j in range(n_var):
        if (random.random() < global_vars.params.p_mut):
            if (ind.xreal[j] == 0):
                ind.xreal[j] = 1
            else:
                ind.xreal[j] = 0
    return

def mutation_pop_bin(pop):
    for p in pop:
        mutation_ind_bin(p)
        
    return

def fill_nondominated_sort(mixed_pop):
    filtered_pop = []
    selected_fronts = assign_rank_crowding_distance(mixed_pop)
    counter = 0
    candidate_fronts = []
    for f in selected_fronts:
        candidate_fronts.append(f)
        counter += len(f)
        if counter > global_vars.params.pop_size:
            break
        
    n_fronts = len(candidate_fronts)

    if n_fronts == 1:
        filtered_pop = []  
    else:
        for i in range(n_fronts - 1):
            filtered_pop.extend(candidate_fronts[i])
    
    n_pop_curr = len(filtered_pop)
    
    sorted_final_front_id = sort_wrt_crowding_dist(candidate_fronts[-1])
    
    
    for i in range(global_vars.params.pop_size - n_pop_curr):
        filtered_pop.append(candidate_fronts[-1][sorted_final_front_id[i]])
    
    

    return filtered_pop
        
def write_final_pop_obj(pop,run):
    f_name = os.path.join('results',global_vars.params.prob_name + '_RUN' + str(run) + str('.out'))
    f = open(f_name,'w')
    for p in pop:
        for i in range(global_vars.params.n_obj):
            f.write('%f \t'%p.fitness[i])
        f.write('\n')
        

def fill_genome(pop):
    encode_pop(pop)

def encode_pop(pop):
    for p in pop:
        encode_ind(p)
    return

def encode_ind(ind):
    genome = []
    n_var_stage = int((global_vars.params.num_operator-1)*(global_vars.params.num_operator)/2 + 1)
    # first loop through number of stages
    for stage in range(global_vars.params.num_stages):
        xreal = ind.xreal[stage*n_var_stage:stage*n_var_stage+n_var_stage]
        genome_stage = [] # this is the encoding within a stage
        # second loop through operator in each stage
        for operator in range(global_vars.params.num_operator-1):
            genome_operator = [] # this is the encoding within a operator
            for element in range(operator+1):
                genome_operator.append(xreal[int(operator*(operator+1)/2 + element)])
            genome_stage.append(genome_operator)
        genome_stage.append([(xreal[-1]).tolist()])
        genome.append(genome_stage)
    ind.genome = genome
    return

def calculate_complexity_ind(ind):
    # first loop through each stage
    operating_nodes = global_vars.params.num_stages*global_vars.params.num_operator
    for stage in range(global_vars.params.num_stages):
        genome_stage = ind.genome[stage]
        for i in range(len(genome_stage)-1):
            if np.sum(genome_stage[i]) == 0:
                non_operating_node = 1
                for j in range(i+1,len(genome_stage)-1):
                    if genome_stage[j][i+1] > 0:
                        non_operating_node = 0
                        break
                operating_nodes = operating_nodes - non_operating_node
        # check for the first operator
        first_operator_redundant = 1
        for k in range(len(genome_stage)-1):
            if genome_stage[k][0] == 1:
                first_operator_redundant = 0
                break
        operating_nodes = operating_nodes - first_operator_redundant
    complexity = operating_nodes
    return complexity

def statistics(pop,archive,run,gen):
    #report the statistics on currect population pop
    accuracy_list = []
    num_child_survived = 0
    for i in range(len(pop)):
        accuracy_list.append(pop[i].fitness[0])
        if pop[i].pt == 0:
            num_child_survived += 1
            pop[i].pt = 1

    meta_statistics = nsga2_classes.MetaStatistics()
    meta_statistics.best_accu = -np.amin(accuracy_list)
    meta_statistics.median_accu = -np.median(accuracy_list)
    meta_statistics.worst_accu = - np.amax(accuracy_list)
    meta_statistics.offspring_survival_rate = num_child_survived / len(pop)
    meta_statistics.archive_size = len(archive)

    print("Best accuracy = {0}, median accuracy = {1}, worst accuracy = {2}.".format(meta_statistics.best_accu,
                                                                                     meta_statistics.median_accu,
                                                                                     meta_statistics.worst_accu))
    print("The offspring survival rate = {0}, archive_size = {1}.".format(meta_statistics.offspring_survival_rate,
                                                                          meta_statistics.archive_size))
    # report to file
    # check if the output save folder exit
    if os.path.exists("output_save"):
        pass
    else:
        os.makedirs("output_save")
    if os.path.exists("output_save/run{}".format(run)):
        pass
    else:
        os.makedirs("output_save/run{}".format(run))
    # 1st save the meta file (best accuracy, median accuracy, worst accuracy,offspring survival rate, archive size
    with open("output_save/run{}/meta_gen{}.pkl".format(run,gen),"wb") as f:
        pickle.dump(meta_statistics,f)

    # 2nd save the population obj, rank, crowdist
    with open("output_save/run{}/pop_gen{}.pkl".format(run,gen),"wb") as f:
        pickle.dump(pop,f)

    # 3rd save the archive
    with open("output_save/run{}/archive_gen{}.pkl".format(run,gen),"wb") as f:
        pickle.dump(archive,f)
