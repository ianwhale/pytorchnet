# -*- coding: utf-8 -*-
"""
Created on Sun Sep 03 17:11:13 2017

@author: Dhebar
"""

#..nsga-ii classes...
import numpy as np

class Parameters:
    def __init__(self):
        self.pop_size = 100
        self.max_gen = 100
        self.n_obj = 2
        self.n_constraints = 0
        self.n_var = 0
        self.Ubound = []
        self.Lbound = []
        self.p_xover = 0.9
        self.p_mut = 0.09
        self.eta_xover = 50
        self.eta_mut = 50
        self.max_runs = 1
        self.prob_name = ''
        self.n_partitions = 1#.....for NSGA-III....
        self.num_stages = 0
        self.num_operator = 0
        self.genome_file_name = ''
        

class Individual:
    def __init__(self):
        self.xreal = np.array([])
        self.genome = []
        self.fitness = np.array([])
        self.crowding_dist = 0.0
        self.rank = -1
        self.S_dom = []
        self.n_dom = -1
        self.address = -1
        self.dir_cosins = np.array([])
        self.tag = 1 # 1 means exactly evaluated from pytorch
        self.pt = 1 # 1 means from parent pop pt, 0 means from child pop qt
        self.parent1_genome = []
        self.parent1_fitness = 0
        self.parent2_genome = []
        self.parent2_fitness = 0
        # GPU_id number for evaluating this individual
        self.gpu_id = 0

class Constants:
    def __init__(self):
        self.EPS = 1.0e-14
        
class IndividualNSGA3(Individual):
    def __init__(self):
        Individual.__init__(self)
        self.normalized_fitness = np.array([])
        self.dist_from_ref_line = -1
        self.ref_line_id = -1
    
class RefLine:
    def __init__(self):
        self.direction = []
        self.niche_count = 0
        self.indivs = []