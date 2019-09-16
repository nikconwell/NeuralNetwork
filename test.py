# Neural network test, as started from https://iamtrask.github.io/2015/07/12/basic-python-network/ but
# changed some things around to make a bit more clear (to me at least)
# Read up on numpy (just past the intro to python which is actually very good)
# http://cs231n.github.io/python-numpy-tutorial/

import numpy as np

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivative(x):
    return x*(1-x)


# Our inputs.  Each set of 3 inputs maps to a single output (see output_dataset)
input_dataset = np.array([  [0,0,1],
                            [0,1,1],
                            [1,0,1],
                            [1,1,1] ])
    
# Outputs we expect for the 4 inputs.            
output_dataset = np.array([ [0],
                            [0],
                            [1],
                            [1]  ])

# seed random numbers to make calculations the same every run - makes comparing runs easier since always get same values
np.random.seed(1)

# Random weights of synapse connections between input and output layer.
# 3 X 1 matrix, numbers ranging from -1 to 1.
#synapse0 = 2*np.random.random((3,1)) - 1
#synapse0 = np.array([[-0.16595599],[0.44064899],[-0.99977125]])
s1 = np.array([[.9],[-.9],[.9]])

# So the idea of the work is that we take the input_dataset, adjust it by the synapse0 weights (multiply)
# and see how close it is to the desired output.  How far off it is from the output is the error and we
# adjust the weights by that derivative of that amount (so a small adjust).  And then repeat for 10,000 times.

file_l1 = open("l1.csv","w+")
#file_s1 = open("s1.csv","w+")

for iter in range(10000):

    # forward propagation, multiply input matrix by weights matrix to figure out layer1 values.
    l1 = sigmoid(np.dot(input_dataset,s1))

    # how much did we miss?
    l1_error = output_dataset - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * derivative(l1)

    # update weights
    s1 += np.dot(input_dataset.T,l1_delta)

    file_l1.write("{}\t{}\t{}\t{}\n".format(l1[0][0],l1[1][0],l1[2][0],l1[3][0]))
#    file_s1.write("{}\t{}\t{}\t{}\n".format(s1[0][0],s1[1][0],s1[2][0],s1[3][0]))

print("Output After Training:")
print(l1)

file_l1.close()
#s1.close()
