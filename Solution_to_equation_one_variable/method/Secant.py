import pandas as pd
import numpy as np
from typing import Callable
def secant(
        f: Callable[[np.float64],np.float64],
        x0: np.float64,
        x1: np.float64,
        xtol: np.float64 = 1.0e-5,
        ftol: np.float64 = 1.0e-5,
        maxiter: np.int64 = 10000
):
    # f is the function in celleble form
    # x0 is the first root 
    # x1 is the second root use to approximate the dericative 
    # f'(x) = f(xn) - f(xn-1) / xn - xn-1
    #ftol: tolerance of abs(f(p))
    #xtol: tolerance of abs(x-x0)
    #maxiter: max iterations
    ar = np.full(shape=(maxiter+1,4), fill_value=np.nan, dtype=np.float64)
    ar[0,:] = [0, x0, x1, f(x0)]
    for i in range(1, maxiter+1):
        ar[i,0] = i
        ar[i,1] = x1
        ar[i, 2] = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        ar[i,3] = f(ar[i,2])
        if abs(ar[i,3]) < ftol or abs(ar[i,2] - x1) < xtol:
                break
                x0 = x1
                x1 = ar[i,2]
        if i == maxiter:
             print("there're too many iterations")
        df = pd.DataFrame(data=ar[0:i+1,:], columns=["iter", "x0", "x1", "f(x1)"], dtype=np.float64)
        return (ar[i,2], df)
if __name__ == "__main__":    
    f = lambda x: np.cos(x) - x
    x, df = secant(f=f, x0=np.pi / 4, x1=np.pi / 3, xtol=-1, ftol=1e-16, maxiter=10000)
    print(df)
    print(f"xn={x:0.6f}")