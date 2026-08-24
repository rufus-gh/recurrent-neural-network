import numpy as np

###### DATA LOADING


with open('data/input.txt', 'r+') as f:
    text = f.read()

    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    data = np.array([char_to_idx[ch] for ch in text])

    f.close()

def chunk_data(data, T):
    index = 0

    n = len(data) // T

    inputs = []
    targets = []

    for i in range(n):

        inp = data[index:(index+T)]
        tar = data[index+1:(index+1+T)]
        inputs.append(inp)
        targets.append(tar)

        index += T

    return np.array(inputs), np.array(targets)

#print(chunk_data(list('abcdefghijklmnopqrstuvwxyz'), 5))
inputs, targets = chunk_data(data, 25)