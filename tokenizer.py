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

for i in range(1000):
# print(pair_counts)
