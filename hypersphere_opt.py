import numpy as np
from scipy import spatial
from scipy.spatial import distance
from scipy.optimize import minimize
from numpy import linalg as LA
import matplotlib.pyplot as plt

def create_hypersphere_loss(num_classes, output_dimension):
    '''
    :param num_classes: refers to K in the paper
    :param output_dimension: refers to D in the paper
    the set of hypersphere should be a matrix of K x D
    '''
    num_classes = 10
    output_dimension = 2
    init_hyperspheres = np.array([np.array(np.random.random(output_dimension)) for x in range(num_classes)]) #init the hyperspheres
    cons = {'type':'eq',
            'fun': l2_norm_constraint,
            'args': (num_classes,output_dimension)}
    res = minimize(cosine_similarity_loss,
                   init_hyperspheres,
                   args = (num_classes, output_dimension),
                   method='SLSQP',
                   constraints = cons,
                   options={'disp': True})
    res_X = np.reshape(res.x, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    print(LA.norm(init_hyperspheres, ord = 2, axis = 0))
    print(LA.norm(res_X, ord = 2, axis = 0))
    plt.scatter(res_X[:, 0], res_X[:, 1])

def cosine_similarity_loss(P, num_classes, output_dimension):
    '''
    The average cosine similarity loss subject to ||X||2 = 1
    '''
    P = np.reshape(P, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    summed_loss = (sum(np.amax((distance.cdist(P, P, 'cosine') * distance.cdist(P, P, 'cosine').T - np.identity(num_classes)), axis = 1)))
    average_loss = (1/num_classes) * summed_loss
    return average_loss

def privilege_info_loss(P, )

def l2_norm_constraint(P, num_classes, output_dimension):
    '''
    Equality constraint for L2 norm
    '''
    P = np.reshape(P, (num_classes, output_dimension)) #scipy auto flattens the hyperspheres, this turns it back to K x D
    l2_norms = LA.norm(P, ord=2, axis=1)
    loss_norm = np.sum(l2_norms) - len(P) #reduce the difference between the sum of norms and the number of numbers
    return loss_norm