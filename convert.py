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
bitarray = np.zeros((len(word)*8,1))

char_index=0
bit_index=0
# Look at each character in the string
for character in word:
    # Look at each bit in the character
    bit_index = 0
    for bit in "{:08b}".format(ord(character)):
        bitarray[(char_index*8)+bit_index][0] = bit
        bit_index += 1
    char_index += 1

print("bitarray:\n",bitarray)

#
# Convert back to characters.
#
index=0
character=0
string=""
for bit in bitarray:
    if (bit[0] == 1):
        character |= (1 << (7 - index))
    index += 1
    if (index == 8):
        string += "{:c}".format(character)
        print(">{}< ({:c})".format(character,character))
        character = 0
        index = 0

print("output: ",string)
