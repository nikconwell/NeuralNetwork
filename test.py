# Neural network test, as started from https://iamtrask.github.io/2015/07/12/basic-python-network/ but
# changed some things around to make a bit more clear (to me at least)
# Read up on numpy (just past the intro to python which is actually very good)
# http://cs231n.github.io/python-numpy-tutorial/

import numpy as np

# Sigmoid function.  Converts any number input to something between 0 and 1.  Has an S shape.  Large -X goes to 0, large X goes to 1.  Center is 0.5
# We use this since we want to get the outputs of our neural network to be either 0 (False) or 1 (True)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivative(x):
    return x*(1-x)


#
# Stuff we are learning.  We need to convert the input_dataset into the
# output dataset.  So we learn the weights or conversion factors that work to convert
# each of the 3 test cases into the output case.
#
# Our inputs.  Each set of 3 inputs maps to a single output (see output_dataset)
input_dataset = np.array([  [0,0,1],
                            [0,1,0],
                            [1,0,0],
                            [1,1,0]])
    
# Outputs we expect for the inputs.            
output_dataset = np.array([ [0],
                            [1],
                            [0],
                            [1]  ])

print("Inputs:")
print(input_dataset)
print()
print("Expected Output:")
print(output_dataset)
print("\n")

#
# Our 3 neuron neural network.  Seed it with some values, doesn't matter what, could be random.  These are the weights
# that we will adjust each iteration until we get closer and closer to the output_dataset values.
#
w1 = np.array([[-.9],[.0],[.9]])

# Could also do random seed:
# np.random.seed(1)
# 3 X 1 matrix, numbers ranging from -1 to 1.
#synapse0 = 2*np.random.random((3,1)) - 1

#
# So the idea of the work is that we take the input_dataset, adjust it by the synapse0 weights (multiply)
# and see how close it is to the desired output.  How far off it is from the output is the error and we
# adjust the weights by that derivative of that amount (so a small adjust).  And then repeat for 10,000 times.
#

#
# We will log our learning findings at each step.
#
file_results = open("results.csv","w+")
file_results.write("iter\tw1\tw2\tw3\tl1\tl2\tl3\tl4\n")

for iter in range(1000):

    # forward propagation, multiply input matrix by weights matrix to get new value.
    input_weighted = (np.dot(input_dataset,w1))
    # Apply a sigmoid function to these weighted values to get them into the 0..1 range since our output_dataset is 0 or 1.  Basically False or True.
    l1 = sigmoid(input_weighted)

    # Figure out how far off from the expected output we were.
    l1_error = output_dataset - l1

    # Figure out how much we will change our weights by.  Too little here and it will take a ton
    # of iterations to converge.  Too much change and we lack resolution and may miss the target
    # by too much and then will adjust back by that.
    l1_delta = l1_error * derivative(l1)

    # update weights
    w1 += np.dot(input_dataset.T,l1_delta)

    #file_l1.write("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(l1[0][0],l1[1][0],l1[2][0],l1[3][0]))
    #file_w1.write("{:.5f}\t{:.5f}\t{:.5f}\n".format(w1[0][0],w1[1][0]))
    
    # Show current state of the learning.
    # Iteration:
    print("\r",end='')
    print("Iteration: {:7,}   ".format(iter),end='')
    print("Weights: ",end='')
    for item in w1:
        print("{:05.5f}  ".format(item[0]),end='')
    print("Learned: ",end='')
    for item in l1:
        print("{:05.5f}  ".format(item[0]),end='')

    file_results.write("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(iter,w1[0][0],w1[1][0],w1[2][0],l1[0][0],l1[1][0],l1[2][0],l1[3][0]))

print()
file_results.close()
