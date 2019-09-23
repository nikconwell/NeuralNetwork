# Neural network test, as started from https://iamtrask.github.io/2015/07/12/basic-python-network/ but
# changed some things around to make a bit more clear (to me at least)
# Read up on numpy (just past the intro to python which is actually very good)
# http://cs231n.github.io/python-numpy-tutorial/

# Update to test.py where we have multiple output bits.  Weights now a 3x3 array,
# 3 for inputs, 3 for outputs.


import numpy as np

import nn as nn


#
# Stuff we are learning.  We need to convert the input_dataset into the
# output dataset.  So we learn the weights or conversion factors that work to convert
# each of the 3 test cases into the output case.
#
# Our inputs.  Each set of 3 inputs maps to a single output (see output_dataset)
input_dataset = np.array([  [0,0,1],
                            [0,1,0],
                            [1,0,0],
                            [1,1,1]])
    
# Outputs we expect for the inputs.            
output_dataset = np.array([ [0,0,0],
                            [1,0,1],
                            [0,1,1],
                            [0,0,0]  ])

print("Inputs:")
print(input_dataset)
print()
print("Expected Output:")
print(output_dataset)
print("\n")

#
# Our neural network of weights.
#

np.random.seed(1)
w1 = 2*np.random.random((3,3))

#
# So the idea of the work is that we take the input_dataset, adjust it by the synapse0 weights (multiply)
# and see how close it is to the desired output.  How far off it is from the output is the error and we
# adjust the weights by that derivative of that amount (so a small adjust).  And then repeat for 10,000 times.
#

#
# We will log our learning findings at each step.  Write column headers of the CSV.
#
#file_results = open("results.csv","w+")
#file_results.write("iter\tw1\tw2\tw3\tl1\tl2\tl3\tl4\n")

for iter in range(10000):

    # forward propagation, multiply input matrix by weights matrix to get new value.
    input_weighted = (np.dot(input_dataset,w1))
    # Apply a sigmoid function to these weighted values to get them into the 0..1 range since our output_dataset is 0 or 1.  Basically False or True.
    l1 = nn.sigmoid(input_weighted)

    # Figure out how far off from the expected output we were.
    l1_error = output_dataset - l1

    # Figure out how much we will change our weights by.  Too little here and it will take a ton
    # of iterations to converge.  Too much change and we lack resolution and may miss the target
    # by too much and then next time we could end up adjusting back by the same amount, thus just oscilating.
    # Note that l1 is 0..1 so derivative(l1) will be 0 to .5, a nice smooth curve like an upside down bowl.  So our multiplication
    # (used later) will be biggest if our value is around .5 (midpoint) but will be very small the closer we get to 0 or 1 meaning that
    # we will make larger updates as we are in the middle (could be true or false, 0 or 1), but will make very small changes as we get
    # closer to 0 or 1.
    # Really the core of this machine learning is searching the solution space looking for better and better answers.
    l1_delta = l1_error * nn.derivative(l1)

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

#    file_results.write("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(iter,w1[0][0],w1[1][0],w1[2][0],l1[0][0],l1[1][0],l1[2][0],l1[3][0]))

print("\n\nOutput learned values (actual)")
for row in l1:
    for column in row:
        print("{:05.5f}  ".format(column),end='')
    print()

print("\nOutput learned values (rounded to T/F)")
for row in l1:
    for column in row:
        print("{:.0f} ".format(column),end='')
    print()

#file_results.close()
