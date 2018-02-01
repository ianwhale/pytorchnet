# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 23:28:59 2017

@author: Dhebar
"""
import nsga2_classes
import math
import global_vars
import numpy as np
import sys
#import evaluate_pytorch
#from ednn.evaluator import Evaluator
#.....problem definitions....

#prob_name = global_vars.params.prob_name
prob_name = 'ednn' 
def compute_fitness_ind(indiv):
    #...bbob...
    if prob_name == 'bbob':
        f1, f2 = global_vars.params.fun(indiv.xreal)
        indiv.fitness[0] = f1
        indiv.fitness[1] = f2
        return
    
    #...ZDT1....
    if prob_name == 'ZDT1':
        f1 = indiv.xreal[0]
        c = np.sum(indiv.xreal[1:])
        g = 1 + 9.0*c/(global_vars.params.n_var - 1)
        f2 = g*(1 - pow((f1*1.0/g),0.5))
        indiv.fitness[0] = f1
        indiv.fitness[1] = f2
        return
    
    if prob_name == 'ZDT2':
        f1 = indiv.xreal[0]
        c = np.sum(indiv.xreal[1:])
        g = 1.0 + 9.0*(c)/(global_vars.params.n_var - 1)
        f2 = g*(1 - math.pow((f1*1.0/g),2))
        
        indiv.fitness[0] = f1
        indiv.fitness[1] = f2
        return
    
    if prob_name == 'ZDT3':
        f1 = indiv.xreal[0]
        c = np.sum(indiv.xreal[1:])
        g = 1.0 + 9.0*(c)/(global_vars.params.n_var - 1)
        f2 = g*(1 - math.pow(f1*1.0/g,0.5) - (f1*1.0/g)*math.sin(10*math.pi*f1))
        indiv.fitness[0] = f1
        indiv.fitness[1] = f2
        return
    
    if prob_name == 'ZDT4':
        f1 = indiv.xreal[0]
        c = 0
        for i in range(1,global_vars.params.n_var):
            c += math.pow(indiv.xreal[i],2) - 10*math.cos(4*math.pi*indiv.xreal[i])
        g = 1 + 10*(global_vars.params.n_var - 1) + c
        f2 = g*(1 - math.sqrt(f1*1.0/g))
        
        indiv.fitness[0] = f1
        indiv.fitness[1] = f2
        
        return
            
    if prob_name == 'ZDT6':
        f1 = 1 - math.exp(-4*indiv.xreal[0])*math.pow(math.sin(6*math.pi*indiv.xreal[0]),6)
        g = 1 + 9.0*math.pow(sum(indiv.xreal[1:])/(global_vars.params.n_var - 1.0),0.25)
        f2 = g*(1 - math.pow(f1*1.0/g,2))
        
        indiv.fitness[0] = f1
        indiv.fitness[1] = f2
        return
    
    if prob_name == 'DTLZ1':
        n_obj = global_vars.params.n_obj
        n_var = global_vars.params.n_var 
        g = 0
        for j in range(n_obj - 1, n_var):
            g += 100.0*(1.0 + math.pow((indiv.xreal[j]-0.5),2) - math.cos(20.0*math.pi*(indiv.xreal[j] - 0.5)))
            
        indiv.fitness[0] = 0.5*indiv.xreal[0]*indiv.xreal[1]*(1+g)
        indiv.fitness[1] = 0.5*indiv.xreal[0]*(1 - indiv.xreal[1])*(1+g)
        indiv.fitness[2] = 0.5*(1 - indiv.xreal[0])*(1 + g)
        
        return
    
    
    if prob_name == 'DTLZ1_g':
        n_obj = global_vars.params.n_obj
        n_var = global_vars.params.n_var 
        g = 0
        for j in range(n_obj - 1, n_var):
            g += 100.0*(1.0 + math.pow((indiv.xreal[j]-0.5),2) - math.cos(20.0*math.pi*(indiv.xreal[j] - 0.5)))
            
        indiv.fitness[0] = 0.5*np.prod(indiv.xreal[:(n_obj - 1)])*(1.0 + g)
        for j in range(1,n_obj-1):
            indiv.fitness[j] = 0.5*np.prod(indiv.xreal[:(n_obj - j - 1)])*(1 - indiv.xreal[n_obj - j])*(1 + g)
            
        indiv.fitness[n_obj - 1] = 0.5*(1 - indiv.xreal[0])*(1 + g)
        
        return
    
    if prob_name == 'DTLZ2':
        n_obj = global_vars.params.n_obj
        n_var = global_vars.params.n_var 
        g = 0
        for j in range(n_obj-1, n_var):
            g = g + ((indiv.xreal[j] - 0.5)*(indiv.xreal[j] - 0.5))
            
        indiv.fitness[0] = (1 + g)*math.cos(indiv.xreal[0]*(math.pi)/2.0)*math.cos(indiv.xreal[1]*(math.pi)/2.0)
        indiv.fitness[1] = (1 + g)*math.cos(indiv.xreal[0]*(math.pi)/2.0)*math.sin(indiv.xreal[1]*(math.pi)/2.0)
        indiv.fitness[2] = (1 + g)*math.sin(indiv.xreal[0]*(math.pi)/2.0)
        return
    
    if prob_name == 'DTLZ2_c':
        n_obj = global_vars.params.n_obj
        n_var = global_vars.params.n_var 
        g = 0
        for j in range(n_obj-1, n_var):
            g = g + ((indiv.xreal[j] - 0.5)*(indiv.xreal[j] - 0.5))
            
        for j in reversed(range(n_obj)):
            prod = 1
            for k in range(n_obj - j):
                prod *= math.cos(indiv.xreal[k]*math.pi/2.0) if (k>=0) else 1.0
            if j == 2:
                indiv.fitness[j] = (1.0+g)*prod*(math.sin(indiv.xreal[n_obj - j -1]*math.pi/2.0) if (j>0) else 1.0)
            else:
                indiv.fitness[j] = (1.0+g)*prod*(math.sin(indiv.xreal[n_obj - j -1]*math.pi/2.0) if (j>0) else 1.0)
                    
    if prob_name == 'ednn':
        #n_obj = global_vars.params.n_obj
        #n_var = global_vars.params.n_var
        #indiv.fitness[0] = np.sum(indiv.xreal)
        #indiv.fitness[1] = evluator.evaluate([inputs])[0]
        #indiv.fitness[1] = -indiv.fitness[1]
        #evaluate_pytorch.demo(indiv.gpu_id)
        fitness1 = 1
        fitness1 = -fitness1
        return (fitness1)
            
    else:
        print('supply problem name')
        sys.exit()

