import numpy as np

###### DATA LOADING


with open('data/input.txt', 'r') as f:
    text = f.read()

    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    data = np.array([char_to_idx[ch] for ch in text])

def chunk_data(data, T):

    n = (len(data)-1) // T

    inputs = []
    targets = []

    for i in range(n):
        index = i*T
        inp = data[index:(index+T)]
        tar = data[index+1:(index+1+T)]
        inputs.append(inp)
        targets.append(tar)

    return np.array(inputs), np.array(targets)

#print(chunk_data(list('abcdefghijklmnopqrstuvwxyz'), 5))
inputs, targets = chunk_data(data, 25)

###### PARAMETER INIT

V=len(chars)
H = 100
W_xh = np.random.randn(H,V) * 0.01
W_hh = np.random.randn(H,H) * 0.01
W_hy = np.random.randn(V,H) * 0.01

b_h = np.zeros(H)
b_y = np.zeros(V)

mem_W_xh = np.zeros((H,V))
mem_W_hh = np.zeros((H,H))
mem_W_hy = np.zeros((V,H))
mem_b_h = np.zeros(H)
mem_b_y = np.zeros(V)