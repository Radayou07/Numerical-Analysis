import numpy as np
def gaussian_elimination(a, b):
    n = len(b)
    # Work on copies to avoid modifying the original matrices
    A = a.astype(np.float32).copy()
    B = b.astype(np.float32).copy()
    
    for k in range(n - 1):
        for i in range(k + 1, n):
            if A[k, k] == 0: continue # Basic check for division by zero
            r = A[i, k] / A[k, k]
            A[i, k] = 0
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - r * A[k, j]
            B[i] = B[i] - r * B[k]
    return A, B

def backward_substitution(u, c):
    n = len(c)
    x = np.zeros(n, dtype=np.float32)
    
    # Start from the last row and go up
    x[n-1] = c[n-1] / u[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum_ax = np.dot(u[i, i+1:], x[i+1:])
        x[i] = (c[i] - sum_ax) / u[i, i]
    return x
# solve the system
a = np.array ([[2,0,1,-1],
              [4,1,3,-2],
              [-1,2,1,2],
              [3,-2,2,1]],dtype=np.float32)
b = np.array([2, 7, 3, 5], dtype=np.float32)
u, c = gaussian_elimination(a, b)
solution = backward_substitution(u, c)
print("The solution is:", solution)