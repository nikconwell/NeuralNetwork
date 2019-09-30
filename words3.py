#!/usr/local/bin/python3

import numpy as np

import nn as nn

from colorama import Fore, Style

import pickle

import sys
import fileinput
import re

#
# Try a 2 layer neural network on words.
#

# What the input to the network is.
input_words = []
output_words = []

# Remind person if they are not feeding in the file
if sys.stdin.isatty():
    print("Expecting lines of input word and output word on stdin.  ^D at end...")

inword = True
for line in fileinput.input():
    line = line.rstrip()
    # Skip blanks and comments
    if (re.match("^\s*$",line.rstrip()) or
        re.match("^\s*#",line.rstrip())):
        next
    if len(line) > nn.__MAXSTRING__:
        print("ERROR - input line too long, max length of {} characters".format(maxstring))
        sys.exit()

    # Flip back and forth adding the word to input_words and output_words
    if (inword == True):
        input_words.append(line)
        inword = False
    else:
        output_words.append(line)
        inword = True



#input_words=["test","word","here","stuf"]
# What we are training to get out of it.
#output_words=["food","outs","abcd","foob"]


# Load up our test words

numwords = len(input_words)
input_bits = np.zeros((numwords,nn.__MAXSTRING__*8))
output_bits = np.zeros((numwords,nn.__MAXSTRING__*8))

for index in range(numwords):
    input_bits[index]=nn.word_to_bits(input_words[index])
    output_bits[index] = nn.word_to_bits(output_words[index])
    index += 1

print("input_bits:")
for row in input_bits:
    for column in row:
        print("{:.0f} ".format(column),end='')
    print()

print("output_bits:")
for row in output_bits:
    for column in row:
        print("{:.0f} ".format(column),end='')
    print()


# Network of weights.  Sized to match input, in bits.  Middle layer 16 bits deep.
np.random.seed(1)
w1 = 2*np.random.random((nn.__MAXSTRING__*8,16))-1
w2 = 2*np.random.random((16,nn.__MAXSTRING__*8))-1

for iter in range(1000):
    l1 = nn.sigmoid(np.dot(input_bits,w1))
    l2 = nn.sigmoid(np.dot(l1,w2))

    l2_error = output_bits - l2
    l2_delta = l2_error*nn.derivative(l2)

    l1_error = l2_delta.dot(w2.T)
    l1_delta = l1_error * nn.derivative(l1)

    w2 += l1.T.dot(l2_delta)
    w1 += input_bits.T.dot(l1_delta)
    
(rows,columns) = np.shape(l2)
print("\n\nOutput learned values (actual)")
for row in range(rows):
    for column in range(columns):
        print("{:05.5f}  ".format(l2[row][column]),end='')
    print()

errors=0
print("\nOutput learned values (rounded to T/F)")
for row in range(rows):
    for column in range(columns):
        formatstring="{:.0f} "
        if (round(l2[row][column]) == output_bits[row][column]):
            formatstring=Fore.GREEN+formatstring+Style.RESET_ALL
        else:
            formatstring=Fore.RED+formatstring+Style.RESET_ALL
            errors += 1
        print(formatstring.format(l2[row][column]),end='')
    print()

print("\n{} errors (differences between input and output)".format(errors))

#
# Save the network to a file.
#
pickle_data=[w1,w2]
network_pickle = open("network.pickle","wb+")
pickle.dump(pickle_data,network_pickle)
network_pickle.close

