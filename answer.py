#!/usr/local/bin/python3

import numpy as np

import nn as nn

from colorama import Fore, Style

import pickle

import sys
import fileinput
import re

# Load in the Neural Network pickle file.

network_pickle = open("network.pickle","rb")
pickle_data = pickle.load(network_pickle)
network_pickle.close

[w1,w2] = pickle_data

if sys.stdin.isatty():
    print("Expecting lines of input (query) words on stdin.  ^D at end.")

for line in fileinput.input():
    line = line.rstrip()
    # Skip blanks and comments
    if (re.match("^\s*$",line) or
        re.match("^\s*#",line)):
        continue
    if len(line) > nn.__MAXSTRING__:
        print("ERROR - input line too long, max length of {} characters".format(maxstring))
        sys.exit()

    input_bits = np.zeros((1,nn.__MAXSTRING__*8))
    input_bits[0] = nn.word_to_bits(line)

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
