# short script to determine a stable state distribution for a Markov chain given a transition matrix

import numpy as np

np.set_printoptions(precision=4, suppress=True)

P =  np.array( [[0.8, 0.1, 0.1],
                [0.0, 0.9, 0.1],
                [0.0, 0.1, 0.9]] )


M = np.eye(3)  # identity matrix
S = P.copy()

for i in range(6):
   S = S @ S
print ("stable check:", np.allclose(S, M))

vals, vecs = np.linalg.eig(P.T)
k = np.argmin(abs(vals - 1))
pi = np.real(vecs[:, k]); pi = pi / pi.sum()
print("eigenvalues:", vals)
print("stationary pi:", pi)