import numpy as np
import pandas as pd
from typing import Callable
def newton(
        f: Callable[[np.float64], np.float64],
        fp: Callable[[np.float64], np.float64],
        x0: np.float64,
        ftol: np.float64 = 1.0e-5,
        xtol: np.float64 = 1.0e-5,
        maxiter: np.int64 = 10000

)->tuple[np.float64,pd.DataFrame]:
    """
    f: the callable function of one parameter
    fp: the callable of derivative function 
    b: upper limit
    ftol: tolerance of abs(f(p))
    xtol: tolerance of abs(x-x0)
    maxiter: max iteration
    """
    ar = np.full(shape=(maxiter+1,2),fill_value=np.nan,dtype=np.float64)
    ar[0,:]=[x0,f(x0)]
    for i in range(1,maxiter+1,1):
        fpx0 = fp(x0)
        if fpx0 == 0:
            print("Newton fail to continue.")
            #cuz we can't devise by 0
            x=x0
            break
        fx0 = f(x0)
        dx = fx0/fpx0
        x = x0 - dx
        fx = f(x)
        ar[i,:]=[x,fx]
        if (abs(x-x0<xtol)|(abs(fx)<ftol)):
            break
        if i == maxiter:
            print("there're too many iter")
        df = pd.DataFrame(data=ar[0:i+1,:],columns=["x","f(x)"],dtype=np.float64)
        return (x,df)

if __name__ == "__main__":
    f = lambda x: np.cos(x) - x
    fp = lambda x: -np.sin(x) - 1
    x, df = newton(f=f, fp=fp, x0=np.pi / 4, xtol=-1, ftol=1e-16, maxiter=10000)
    print(df)
    print(f"xn={x:0.6f}")