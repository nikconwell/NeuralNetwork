#!/usr/local/bin/python3

import numpy as np

import nn as nn

from colorama import Fore, Style

import pickle

# Try loading a pickle file.

#pickle_data=[w1,w2]
network_pickle = open("network.pickle","rb")
pickle_data = pickle.load(network_pickle)
network_pickle.close

[w1,w2] = pickle_data

# What the input to the network is.
#input_words=["test","word","here","stuf"]
input_words=["test"]

numwords = len(input_words)
input_bits = np.zeros((numwords,nn.__MAXSTRING__*8))

for index in range(numwords):
    input_bits[index]=nn.word_to_bits(input_words[index])
    index += 1

#
# Calculate outputs based on input and network weights
#
l1 = nn.sigmoid(np.dot(input_bits,w1))
l2 = nn.sigmoid(np.dot(l1,w2))

(rows,columns) = np.shape(l2)
#print("\nOutput learned values (rounded to T/F)")
#for row in range(rows):
#    for column in range(columns):
#        formatstring="{:.0f} "
#        print(formatstring.format(l2[row][column]),end='')
#    print()
#print("\nString format:")

#
# Convert binary back to characters
#
index=0
character=0
string=""
for row in range(rows):
    for column in range(columns):
        if (round(l2[row][column]) == 1):
            character |= (1 << (7 - index))
        index += 1
        if (index == 8):
            string += "{:c}".format(character)
            #print(">{}< ({:c})".format(character,character))
            character=0
            index=0
print("output: ",string)
