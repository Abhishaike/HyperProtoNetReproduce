import numpy as np
from scipy import spatial
from scipy.spatial import distance
from scipy.optimize import minimize
from numpy import linalg as LA
import matplotlib.pyplot as plt

def create_hypersphere_loss(num_classes, output_dimension, unique_classes, unique_class_numbers, use_privileged_info = None):
    '''
    :param num_classes: refers to K in the paper
    :param output_dimension: refers to D in the paper
    :param unique_classes: used to create embedding vectors to better inform the optimization, not used right now
    the set of hypersphere should be a matrix of K x D
    '''
    init_hyperspheres = np.array([np.array(np.random.random(output_dimension)) for x in range(num_classes)]) #init the hyperspheres
    cons = {'type':'eq',
            'fun': l2_norm_constraint,
            'args': (num_classes,output_dimension)} #l2 norm equality constraint
    if use_privileged_info is False: #if there is no priv. information, just use cosine dist to distribute the points
        res = minimize(cosine_similarity_loss,
                       init_hyperspheres,
                       args=(num_classes, output_dimension),
                       method='SLSQP',
                       constraints=cons,
                       options={'disp': True, 'maxiter': 5})
    else:
        res = minimize(combined_loss, #else, use the sum of the cosine and priv. info to distribute points
                       init_hyperspheres,
                       args = (num_classes, output_dimension, unique_classes),
                       method='SLSQP',
                       constraints = cons,
                       options={'disp': True, 'maxiter': 5})
    optimized_points = np.reshape(res.x, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    class_matched_points = dict(zip(unique_class_numbers, optimized_points))
    return class_matched_points
    # print(LA.norm(init_hyperspheres, ord = 2, axis = 0))
    # print(LA.norm(optimized_points, ord = 2, axis = 0))
    # plt.scatter(res_X[:, 0], res_X[:, 1])

def cosine_similarity_loss(P, num_classes, output_dimension):
    '''
    The average cosine similarity loss subject to ||X||2 = 1
    '''
    P = np.reshape(P, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    summed_loss = (sum(np.amax((distance.cdist(P, P, 'cosine') * distance.cdist(P, P, 'cosine').T - np.identity(num_classes)), axis = 1)))
    average_loss = (1/num_classes) * summed_loss
    return average_loss

def l2_norm_constraint(P, num_classes, output_dimension):
    '''
    Equality constraint for L2 norm
    '''
    P = np.reshape(P, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    l2_norms = LA.norm(P, ord=2, axis=1)
    loss_norm = np.sum(l2_norms) - len(P) #reduce the difference between the sum of norms and the number of numbers
    return loss_norm

def privilege_info_loss(P, num_classes, output_dimension):
    '''
    Not sure how to use this, the paper didn't specify to get the embedding info?
    It doesn't seem to be useful anyway once D gets reasonably large, so it's fine
    '''
    return 0

def combined_loss(P, num_classes, output_dimension):
    cosine_similarity = cosine_similarity_loss(P, num_classes, output_dimension)
    privilege_info = privilege_info_loss(P, num_classes, output_dimension)
    return cosine_similarity + privilege_info