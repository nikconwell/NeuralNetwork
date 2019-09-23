#!/usr/local/bin/python3

import numpy as np

import nn as nn

#
# Try a single layer neural network on words.
#

# What the input to the network is.
#input_words=["test","word","here","stuf"]
# What we are training to get out of it.
#output_words=["food","outs","abcd","foob"]

#
# Convert words to an array
#

def word_to_bits(word):
    bits = np.zeros((len(word)*8))
    char_index=0
    for character in word:
        bit_index=0
        print (">{}< ({:08b})".format(character,ord(character)))
        for bit in "{:08b}".format(ord(character)):
            bits[(char_index*8)+bit_index] = bit
            bit_index += 1
        char_index += 1
    return(bits)

input_bits = np.zeros((1,32))
input_bits[0]=word_to_bits("test")
print("input_bits:\n",input_bits)

output_bits = np.zeros((1,32))
output_bits=word_to_bits("food")
print("output_bits:\n",output_bits)

# Network of weights.  32 to match the input, 32 to match the output
np.random.seed(1)
w1 = 2*np.random.random((32,32))-1

for iter in range(1000):
    l1 = nn.sigmoid(np.dot(input_bits,w1))
        
    l1_error = output_bits - l1
    l1_delta = l1_error * nn.derivative(l1)

    w1 += np.dot(input_bits.T,l1_delta)

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