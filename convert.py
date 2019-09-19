#!/usr/local/bin/python3

import numpy as np

#
# Take a string and convert to a bit array.  Useful for converting things into something we can learn on.
#

word = "test"
print("input: ",word)
for character in word:
    print (">{}< ({} - {:08b})".format(character,ord(character),ord(character)))

# Create a bit array based on length of word * 8 for 8 bits per character.
bitarray = np.zeros(len(word)*8)

index=0
# Look at each character in the string
for character in word:
    # Look at each bit in the character
    for bit in "{:08b}".format(ord(character)):
        bitarray[index] = bit
        index += 1

print("bitarray:\n",bitarray)

#
# Convert back to characters.
#
index=0
character=0
string=""
for bit in bitarray:
    if (bit == 1):
        character |= (1 << (7 - index))
    index += 1
    if (index == 8):
        print(">{}< ({:c})".format(character,character))
        string += "{:c}".format(character)
        character = 0
        index = 0

print("output: ",string)