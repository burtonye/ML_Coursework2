#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:38:07 2018

Performing bagging of decision trees

@author: E Robinson
"""

import DecisionTree as DT
import numpy as np

def bootstrap_sample(dataset,random_state=42):
    ''' Create a random subsample from the dataset with replacement
        input:
            dataset: (n_samples,n_features) data array
            random_state: fixes random seed
        output:
            samples: array of data bootstrapped from dataset
    '''
    n_sample = dataset.shape[0] 
    indices=np.random.choice(dataset.shape[0], n_sample,replace=True)
    
    samples=[]
    for i in indices:
        samples.append(dataset[i])
    
    return np.asarray(samples)

def create_bagged_ensemble(data,max_depth, min_size, n_trees,random_state=42):
    
    ''' Create a bagged ensemble of decision trees
    input:
        data: (n_samples,n_features) data array
        max_depth: max depth of trees
        min_size: minimum number of samples allowed in tree leaf nodes
        n_trees: total number of trees in the ensemble
        random_state: fixes random seed
    output:
        bagged_ensemble: list of decision trees that make up the bagged ensemble
    '''
    
    bagged_ensemble=[]
    for i in range(n_trees):
        sample = bootstrap_sample(data,random_state)
        tree = DT.build_tree(sample, max_depth, min_size)
        bagged_ensemble.append(tree)
    
    return bagged_ensemble
        
def bagging_predict(trees, testdata):
    
    predictions=[]
    
    for row in testdata:
        row_predictions=[]

        for  tree in trees: 
            row_predictions.append(DT.predict_row(tree, row))
        count=np.bincount(row_predictions)
        predictions.append(np.argmax(count))
            
    return predictions


# =============================================================================
# 
# ratio=0.5
# 
# 
# DATA_population=np.random.randint(0,20,100000)
# print(DATA_population.shape)
# 
# print('The mean of the population is {} '.format(np.mean(DATA_population)))   
# # print('The mean and std of the sample are {} {}'.format(np.mean(DATA_sample),np.std(DATA_sample)))   
# 
# 
# # taking varying numbers of bootstrap samples of size fraction 0.1
# #for sample_size in [1, 100, 1000, 5000]:
# 
# sample_size=1
# DATA_sample=np.random.choice(DATA_population, sample_size)
# for n_samples in [1, 100, 1000,5000,10000]:
#     sample_means = []; population_means = []
#     for i in range(n_samples):
#         population_sample = np.random.choice(DATA_population, sample_size)
#         sample = subsample(DATA_sample, 1)
#         sample_means.append(np.mean(sample))
#         population_means.append(np.mean(population_sample))
#     print('Samples={}, Estimated Population Mean: {}'.format(n_samples, np.mean(population_means)))
#     print('Samples={}, Estimated Mean: {}'.format(n_samples, np.mean(sample_means)))
# 
# 
# =============================================================================
