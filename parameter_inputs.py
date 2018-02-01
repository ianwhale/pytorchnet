# -*- coding: utf-8 -*-
"""
Created on Sun Sep 03 17:26:44 2017

@author: Dhebar
"""
import nsga2_classes
import numpy as np

##....parameter inputs....


#def input_parameters():
#    params = nsga2_classes.Parameters()
#    params.pop_size = 92
#    params.max_gen = 400
#    params.n_obj = 3
#    params.n_constraints = 0
#    params.n_var = 7
#    params.Lbound = np.array([0.0]*params.n_var)
#    params.Ubound = np.array([1.0]*params.n_var)
#    params.p_xover = 1.0
#    params.p_mut = 1.0/params.n_var
#    params.eta_xover = 30.0
#    params.eta_mut = 20.0
#    params.max_runs = 1
#    params.prob_name = 'DTLZ1'
#    params.n_partitions = 12
#    return params


def input_parameters():
    params = nsga2_classes.Parameters()
    params.pop_size = 20
    params.max_gen = 50
    params.n_obj = 2
    params.n_constraints = 0
    params.n_var = 15
    params.Lbound = np.array([0.0]*params.n_var)
    params.Ubound = np.array([1.0]*params.n_var)
    params.p_xover = 1.0
    params.p_mut = 0.1
    params.eta_xover = 30
    params.eta_mut = 20
    params.max_runs = 1
    params.prob_name = "ednn"
    params.n_partitions = 12
    params.num_stages = 3  # this is how many in between blocks
    params.num_operator = 4 # this is how many operator per stage
    params.strategy = 1
    return params
    
def input_constants():
    cons = nsga2_classes.Constants()
    cons.EPS = 1.0e-14
    return cons
