import numpy as np

def ldl_decomposition(a):
    n = a.shape[0]
    L = np.eye(n, dtype=np.float32)
    D = np.zeros((n, n), dtype=np.float32)

    for j in range(n):
        sum_d = sum(L[j, k]**2 * D[k, k] for k in range(j))
        D[j, j] = a[j, j] - sum_d

        if abs(D[j, j]) < 1e-12:
            raise ValueError(f"Zero pivot at D[{j},{j}]")

        for i in range(j + 1, n):
            sum_l = sum(L[i, k] * L[j, k] * D[k, k] for k in range(j))
            L[i, j] = (a[i, j] - sum_l) / D[j, j]

    return L, D

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def backward_substitution(U, c):
    n = len(c)
    x = np.zeros(n, dtype=np.float32)
    x[n-1] = c[n-1] / U[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (c[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

# Symmetric matrix
a = np.array([[ 2,  4, -1,  3],
              [ 4,  1,  2, -2],
              [-1,  2,  1,  2],
              [ 3, -2,  2,  1]], dtype=np.float32)
b = np.array([2, 7, 3, 5], dtype=np.float32)

L, D = ldl_decomposition(a)
y = forward_substitution(L, b)
z = y / np.diag(D)           # cleaner than a loop
solution = backward_substitution(L.T, z)

print("The solution is:", solution)

# Verify: a @ solution should equal b
print("Residual:", np.allclose(a @ solution, b))