import numpy as np
from scipy.special import softmax

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

T = 25
#print(chunk_data(list('abcdefghijklmnopqrstuvwxyz'), 5))
inputs, targets = chunk_data(data, T)

###### PARAMETER INIT

V=len(chars)
H = 256

layers = 0

W_xh = [np.random.randn(H,V) * 0.01 for _ in range(layers)]
W_hh = [np.random.randn(H,H) * 0.01 for _ in range(layers)]
W_hy = np.random.randn(V,H) * 0.01

b_h = [np.zeros(H) for _ in range(layers)]
b_y = np.zeros(V)

mem_W_xh = np.zeros((H,V))
mem_W_hh = np.zeros((H,H))
mem_W_hy = np.zeros((V,H))
mem_b_h = np.zeros(H)
mem_b_y = np.zeros(V)

###### FORWARD PASS

def one_hot_encode(char):
    temp = np.zeros(V)
    temp[char] = 1
    return temp

def forward_pass(input_chunk, target_chunk, h_prev):
    h = [np.empty((T+1, H)) for _ in range(layers)]
    
    L = 0
    x = np.empty((T, V))
    p = np.empty((T, V))
    for i in range(layers):
        h[i][0] = h_prev[i]
    for t in range(T):
        x[t] = one_hot_encode(input_chunk[t])
        layer_input = x[t]
        for i in range(layers):
            h[i][t+1] = np.tanh((W_xh[i] @ layer_input) + (W_hh[i] @ h[i][t]) + b_h[i])
            layer_input = h[i][t+1]
        y = W_hy @ h[layers - 1][t+1] + b_y
        y_shifted = np.exp(y - np.max(y))
        p[t] = y_shifted / np.sum(y_shifted)
        L += -1 * np.log(p[t][target_chunk[t]])

    h_last = [h[i][T] for i in range(layers)]
    return L, x, h, p, h_last

#L, x, h_cache, p, h_last = forward_pass(inputs[0], targets[0], np.zeros(H))

############ BACKWARD PASS

def backward_pass(x, h_cache, p, target_chunk):
    dW_xh = np.zeros((H,V))
    dW_hh = np.zeros((H,H))
    dW_hy = np.zeros((V,H))
    db_h = np.zeros(H)
    db_y = np.zeros(V)

    dh_next = np.zeros(H)
    dh = np.zeros((T, H))

    for t in range(T-1, -1, -1):
        dy = p[t].copy()
        dy[target_chunk[t]] -= 1
        dW_hy += np.outer(dy, h_cache[t+1])
        db_y += dy
        dh_t = np.transpose(W_hy) @ dy + dh_next
        dz = dh_t * (1 - h_cache[t+1] * h_cache[t+1])
        dW_xh += np.outer(dz, x[t])
        dW_hh += np.outer(dz, h_cache[t])
        db_h += dz
        dh_next = np.transpose(W_hh) @ dz

    return np.clip(dW_xh, -5, 5),\
           np.clip(dW_hh, -5, 5),\
           np.clip(dW_hy, -5, 5),\
           np.clip(db_h, -5, 5),\
           np.clip(db_y, -5, 5)

#dW_xh, dW_hh, dW_hy, db_h, db_y = backward_pass(x, h_cache, p, targets[0])

########## ADAGRAD 

def adagrad(dW_xh, dW_hh, dW_hy, db_h, db_y):
    global W_xh, W_hh, W_hy, b_h, b_y,mem_W_xh, mem_W_hh, mem_W_hy, mem_b_h, mem_b_y
    
    mem_W_xh += dW_xh ** 2
    mem_W_hh += dW_hh ** 2
    mem_W_hy += dW_hy ** 2
    mem_b_h  += db_h  ** 2
    mem_b_y  += db_y  ** 2

    learning_rate = 0.1
    epsilon = 1 * (10 ** (-8))

    W_xh -= learning_rate * dW_xh / (np.sqrt(mem_W_xh) + epsilon)
    W_hh -= learning_rate * dW_hh / (np.sqrt(mem_W_hh) + epsilon)
    W_hy -= learning_rate * dW_hy / (np.sqrt(mem_W_hy) + epsilon)
    b_h -= learning_rate * db_h / (np.sqrt(mem_b_h) + epsilon)
    b_y -= learning_rate * db_y / (np.sqrt(mem_b_y) + epsilon)

#################### SAVE THE DATA

def save_model(path, W_xh, W_hh, W_hy, b_h, b_y, char_to_idx, idx_to_char):
    np.savez(path,
             W_xh=W_xh, W_hh=W_hh, W_hy=W_hy,
             b_h=b_h, b_y=b_y,
             chars=np.array(list(char_to_idx.keys())))

######### TRAINING LOOP

import matplotlib.pyplot as plt

smooth_loss = -np.log(1.0/V) * T   # reasonable init value: expected loss at random init
loss_history = []

h_prev = np.zeros(H)
i = 0
num_iterations = 1000000 # or however long you want to train
'''
for iteration in range(num_iterations):
    if i == 0:
        h_prev = np.zeros(H)

    L, x, h_cache, p, h_last = forward_pass(inputs[i], targets[i], h_prev)
    dW_xh, dW_hh, dW_hy, db_h, db_y = backward_pass(x, h_cache, p, targets[i])
    adagrad(dW_xh, dW_hh, dW_hy, db_h, db_y)
    h_prev = h_last

    smooth_loss = smooth_loss * 0.999 + L * 0.001
    if iteration % 100 == 0:
        loss_history.append(smooth_loss)
        print(f"iter {iteration}, loss {smooth_loss:.4f}")

    if iteration % 150000 == 0 and iteration > 0:
        save_model(f'checkpoint_H256_{iteration}.npz', W_xh, W_hh, W_hy, b_h, b_y, char_to_idx, idx_to_char)

    i += 1
    if i >= len(inputs):
        i = 0

save_model('model_checkpoint.npz', W_xh, W_hh, W_hy, b_h, b_y, char_to_idx, idx_to_char)

plt.plot(loss_history)
plt.xlabel('iteration (x100)')
plt.ylabel('smoothed loss')
plt.title('Training loss')
plt.show()

print("Finished", num_iterations, " iterations.")
'''
#### LOAD DATA

def load_model(path):
    data = np.load(path)
    W_xh = data['W_xh']
    W_hh = data['W_hh']
    W_hy = data['W_hy']
    b_h = data['b_h']
    b_y = data['b_y']

    chars = sorted(data['chars'].tolist())
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    return W_xh, W_hh, W_hy, b_h, b_y, char_to_idx, idx_to_char

W_xh, W_hh, W_hy, b_h, b_y, char_to_idx, idx_to_char = load_model('checkpoint_H256_10950000.npz')

####### RESULTS

seed_char = 'a'

x = one_hot_encode(char_to_idx[seed_char])
h = h_prev
characters = 5000
output = []

for i in range(characters):
    h = np.tanh(W_xh @ x + W_hh @ h + b_h)
    y = W_hy @ h + b_y
    p = softmax(y)
    new_char = np.random.choice(chars, p=p)
    output.append(str(new_char))
    x = one_hot_encode(char_to_idx[new_char])

print(seed_char + ''.join(output))