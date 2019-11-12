import numpy as np
from scipy import spatial
from scipy.spatial import distance
from scipy.optimize import minimize
from numpy import linalg as LA
import matplotlib.pyplot as plt
import torch
from numba import jit, float64, int64
import scipy

def create_hypersphere_loss_wo_constraints(num_classes, output_dimension, unique_classes, unique_class_numbers, epochs = 500, use_privileged_info=None):
    '''
    :param num_classes: refers to K in the paper
    :param output_dimension: refers to D in the paper
    :param unique_classes: used to create embedding vectors to better inform the optimization, not used right now
    the set of hypersphere should be a matrix of K x D
    '''
    np.random.seed(seed=0)
    init_hyperspheres = np.array([-2*np.array(np.random.random(output_dimension))+1 for x in range(num_classes)]) #init the hyperspheres
    optimized_hyperspheres = (init_hyperspheres.T / np.linalg.norm(init_hyperspheres, axis=1)).T
    if use_privileged_info is False:  # if there is no priv. information, just use cosine dist to distribute the points
        for repeats in range(0, epochs): #optimize for one step, project into l2, and repeat optimization
            optimized_hyperspheres = (optimized_hyperspheres.T / np.linalg.norm(optimized_hyperspheres, axis=1)).T
            res = minimize(cosine_similarity_loss,
                           optimized_hyperspheres,
                           args=(num_classes, output_dimension),
                           method='BFGS',
                           options={'disp': None, 'maxiter': 10})
            optimized_hyperspheres = np.reshape(res.x, (num_classes, output_dimension))  # scipy auto flattens the hyperspheres, this turns it back to K x D
            print('Hypersphere init loss:', res.fun)
    else:
        for repeats in range(0, epochs):
            optimized_hyperspheres = (optimized_hyperspheres.T / np.linalg.norm(optimized_hyperspheres, axis=1)).T
            res = minimize(combined_loss,
                           optimized_hyperspheres,
                           args=(num_classes, output_dimension, unique_classes),
                           method='BFGS',
                           options={'disp': None, 'maxiter': 10})
            optimized_hyperspheres = np.reshape(res.x, (num_classes, output_dimension))  # scipy auto flattens the hyperspheres, this turns it back to K x D
            print('Hypersphere init loss:', res.fun)

    optimized_hyperspheres = (optimized_hyperspheres.T / np.linalg.norm(optimized_hyperspheres, axis=1)).T
    class_matched_points = dict(zip(unique_class_numbers, list(optimized_hyperspheres))) #match up the class matched points
    return class_matched_points
    # print(LA.norm(init_hyperspheres, ord = 2, axis = 1))
    # print(LA.norm(optimized_points, ord = 2, axis = 1))
    # plt.scatter(res_X[:, 0], res_X[:, 1])


def create_hypersphere_loss_w_constraints(num_classes, output_dimension, unique_classes, unique_class_numbers, use_privileged_info = None):
    '''
    :param num_classes: refers to K in the paper
    :param output_dimension: refers to D in the paper
    :param unique_classes: used to create embedding vectors to better inform the optimization, not used right now
    the set of hypersphere should be a matrix of K x D

    IDENTICAL TO THE ABOVE LOSS, EXCEPT INBUILT CONSTRAINTS ARE USED AND L2 REPROJECTION IS ONLY DONE AT THE END OF THE
    OPTIMIZATION. NOT USED DUE TO EXTREMELY LARGE TIME DEMANDS.
    '''
    np.random.seed(seed=0)
    init_hyperspheres = np.array([-2*np.array(np.random.random(output_dimension))+1 for x in range(num_classes)]) #init the hyperspheres
    init_hyperspheres = (init_hyperspheres.T / np.linalg.norm(init_hyperspheres, axis=1)).T
    cons = {'type':'eq',
            'fun': l2_norm_constraint,
            'args': (num_classes,output_dimension)} #l2 norm equality constraint
    if use_privileged_info is False: #if there is no priv. information, just use cosine dist to distribute the points
        res = minimize(cosine_similarity_loss,
                       init_hyperspheres,
                       args=(num_classes, output_dimension),
                       method='SLSQP',
                       constraints=cons,
                       options={'disp': True, 'maxiter': 50})
    else:
        res = minimize(combined_loss, #else, use the sum of the cosine and priv. info to distribute points
                       init_hyperspheres,
                       args = (num_classes, output_dimension, unique_classes),
                       method='SLSQP',
                       constraints = cons,
                       options={'disp': True, 'maxiter': 5})
    optimized_points = np.reshape(res.x, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    norm_optimized_points = (optimized_points.T / np.linalg.norm(optimized_points, axis=1)).T
    class_matched_points = dict(zip(unique_class_numbers, norm_optimized_points))
    return class_matched_points
    # print(LA.norm(init_hyperspheres, ord = 2, axis = 1))
    # print(LA.norm(optimized_points, ord = 2, axis = 1))
    # plt.scatter(res_X[:, 0], res_X[:, 1])

def cosine_similarity_loss(P, num_classes, output_dimension):
    '''
    The average cosine similarity loss subject to ||X||2 = 1
    '''
    P = np.reshape(P, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    P = (P.T / np.linalg.norm(P, axis=1)).T
    summed_loss = np.sum(np.amax(np.matmul(P, P.T) + 1 - 2 * np.identity(num_classes), axis = 1))
    average_loss = (1/num_classes) * summed_loss
    return average_loss

def l2_norm_constraint(P, num_classes, output_dimension):
    '''
    Equality constraint for L2 norm. Used in create_hypersphere_loss_w_constraints
    '''
    P = np.reshape(P, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    l2_norms = LA.norm(P, ord=2, axis=1)
    loss_norm = np.sum(np.abs(l2_norms - np.ones(num_classes))) #reduce the difference between the sum of norms and the number of numbers
    return loss_norm

def privilege_info_loss(P, num_classes, output_dimension, unique_classes):
    '''
    Insert the loss function for the priv info here.
    '''
    return 0

def combined_loss(P, num_classes, output_dimension, unique_classes):
    cosine_similarity = cosine_similarity_loss(P, num_classes, output_dimension)
    privilege_info = privilege_info_loss(P, num_classes, output_dimension, unique_classes)
    return cosine_similarity + privilege_info