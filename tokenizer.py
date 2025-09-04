# import torch
import time
from collections import Counter

# fundamentally, need a list of pair replacements.

# They go in order. first one only considers chars, second one considers chars plus first token, etc.

# calculate stats


with open('input.txt', 'r') as file:
    chars = list(file.read().encode('ascii'))

pair_counts = Counter(zip(chars, chars[1:]))

pair_substitutions = []

# for i in range(1000):
pair = max(Counter(zip(chars, chars[1:])))
# print(Counter(zip(chars, chars[1:])))
print('\n'.join(f'{chr(key[0])}{chr(key[1])}: {value}' for key, value in sorted(Counter(zip(chars, chars[1:])).items(), key=lambda x: x[1])))
print(''.join(chr(item) for item in pair))
    # take the most common pairing and replace it
