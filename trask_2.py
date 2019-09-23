
#
# From https://iamtrask.github.io/2015/07/12/basic-python-network/
#
# Problem with this code is that if the input is all [0,0,0] the weights
# are going to top out at 0.5 (based on sigmoid function) since we multiply
# the input by the weights, and input of 0 will always multiply out at 0.
# I'm thinking the input needs to add or we need an additional layer or
# something that can go to 1 and we use that to trigger, otherwise we can
# never get a 1 output on a 0 input, no matter what weight we multiply it
# by.
#

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,0],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[1,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print("Output After Training:")
print(l1)
