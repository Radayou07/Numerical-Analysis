import numpy as np
def lu_decomposition(a):
    n=a.shape[0]
    l=np.eye(N=n)
    u=a.copy()
    for k in range (n-1):
        for i in range (k+1,n):
            l[i,k]=u[i,k]/u[k,k]
            u[i,k]=0
            for j in range (k+1,n):
                u[i,j]=u[i,j]-l[i,k]*u[k,j]
    return l,u
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sum_ly = np.dot(L[i, :i], y[:i])
        y[i] = (b[i] - sum_ly) / L[i, i]
    return y
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
a = np.array([[ 2,  4, -1,  3],
              [ 4,  1,  2, -2],
              [-1,  2,  1,  2],
              [ 3, -2,  2,  1]], dtype=np.float32)
b = np.array([2, 7, 3, 5], dtype=np.float32)
l,u=lu_decomposition(a)
y = forward_substitution(l, b)
solution = backward_substitution(u, y)
print("The solution is:", solution)