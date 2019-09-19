import numpy as np

# Sigmoid function.  Converts any number input to something between 0 and 1.  Has an S shape.  Large -X goes to 0, large X goes to 1.  Center is 0.5
# We use this since we want to get the outputs of our neural network to be either 0 (False) or 1 (True)
def sigmoid(x):
    return 1/(1+np.exp(-x))

#
# Used to calculate how much adjustment we will do.  Larger adjustments if we are near the midpoint of 0..1, very small adjustment the closer we are to
# 0 or 1.
def derivative(x):
    return x*(1-x)
