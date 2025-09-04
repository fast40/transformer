import pickle
from collections import Counter
import pathlib

from tqdm import tqdm


def merge(x, new_token, new_token_id):
    new_tokens = []
    i = 0
    while i < len(x):
        if i+1 < len(x) and x[i] == new_token[0] and x[i+1] == new_token[1]:
            new_tokens.append(new_token_id)
            i += 2
        else:
            new_tokens.append(x[i])
            i += 1
    return new_tokens


def make_tokens(chars):
    tokens = [[i] for i in range(128)]  # add the og chars

    for new_token in tqdm(range(2048 - 128)):
        pair, occurrances = max(Counter(zip(chars, chars[1:])).items(), key=lambda x: x[1])

        if occurrances == 1:
            break

        tokens.append([*tokens[pair[0]], *tokens[pair[1]]])

        chars = merge(chars, pair, new_token + 128)
    
    return tokens


def main():
    with open('input.txt', 'r') as file:
        chars = list(file.read().encode('ascii'))

    tokens = make_tokens(chars)

    with open('tokens.bin', 'wb') as file:
        print('saving')
        pickle.dump(tokens, file)
        print('saved')


if __name__ == '__main__':
    if pathlib.Path('tokens.bin').is_file():
        with open('tokens.bin', 'rb') as file:
            tokens = pickle.load(file)
            print('\n'.join(repr(''.join(chr(c) for c in token)) for token in tokens))
            t = [''.join(chr(c) for c in token) for token in tokens]
            print('\n'.join(repr(x) for x in sorted(t)))
            # print('RIAR LAUR' in t)
        print('tokens.bin already exists. quitting.')
        quit()

    main()
