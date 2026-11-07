import numpy as np

def server_win_prob(p):
    q = 1 - p
    P = np.array([[0, p, q, 0, 0],
                  [q, 0, 0, p, 0],
                  [p, 0, 0, 0, q],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    return np.linalg.matrix_power(P, 100)[0, 3]   # start at D, end at WS

for p in (0.5, 0.6, 0.7):
    print(p, server_win_prob(p), p**2 / (p**2 + (1-p)**2))  # p^2 / (p^2 + q^2)