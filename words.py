#!/usr/local/bin/python3

import numpy as np

import nn as nn

#
# Try a neural network on words.
#

# What the input to the network is.
input_word="test"
# What we are training to get out of it.
output_word="food"

#
# Convert words to an array
#

def word_to_bits(word):
    bits = np.zeros((len(word)*8,1))
    char_index=0
    for character in word:
        bit_index=0
        print (">{}< ({:08b})".format(character,ord(character)))
        for bit in "{:08b}".format(ord(character)):
            bits[(char_index*8)+bit_index][0] = bit
            bit_index += 1
        char_index += 1
    return(bits)

input_bits = word_to_bits(input_word)
print("input_bits:\n",input_bits)

output_bits = word_to_bits(output_word)

# Network of weights.  32 bits to match the 4 bytes of 8 bits of input.

w1 = np.zeros((1,32))

for iter in range(10000):
    input_weighted = (np.dot(input_bits,w1))
    l1 = nn.sigmoid(input_weighted)
    l1_error = output_bits - l1
    l1_delta = l1_error * nn.derivative(l1)
    w1 += np.dot(input_bits.T,l1_delta)

print(l1)
