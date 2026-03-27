import numpy as np
import pandas as pd
from typing import Callable
def bisection(
        f: Callable[[np.float64], np.float64],
        a: np.float64,
        b: np.float64,
        ftol: np.float64 = 1.0e-5,
        xtol: np.float64 = 1.0e-5,
        maxiter: np.int64 = 10000

)->tuple[np.float64,pd.DataFrame]:
    """
    f: the callable function of one parameter
    a: lower limit
    b: upper limit
    ftol: tolerance of abs(f(p))
    xtol: tolerance of abs(p-a)
    maxiter: max iteration
    p: the approximate answer
    """
    ar =np.full(shape=(maxiter+1,4),fill_value=np.nan,dtype = np.float64)
    #this is the box for the answer with maxiter+1 row and 4 column for a , b , p , f(P)
    ar[0,:]=[a,b,np.nan,np.nan]
    #the first row =[a,b,0,0]
    for i in  range (1,maxiter+1,1):
        # for loop  of i in range of 1 and maxiter +1 with the step size of 1
        fa = f(a)
        #make a new variable = the value of function on the lower limit
        p = 0.5*(a+b)
        #p is the half point of a and b
        fp = f(p)
        #fine the value of f(p)
        ar[i,:]=[a,b,p,fp]
        #put answer in its box
        if (abs(p-a)<xtol)|(abs(fp)<ftol):
            #check if the answer is ready
            break
        if (fa * fp > 0):
            a = p
        else:
            b = p  # You need this to narrow the interval from the other side!
        if i == maxiter:
            print('there are way too many iteration.')
        df = pd.DataFrame(
            data=ar[0:i+1:],
            #make a datafrome to store the answer from the first iteration to the one we're on 
            columns = [a,b,p,f(p)],
            dtype =np.float64
        )
    return (p,df)
if __name__ == "__main__":
    def f(x: np.float64) -> np.float64:
        return x**3 - 4 * x + 2

    p, df = bisection(f=f, a=0, b=1, xtol=-1, ftol=-1, maxiter=25)
    print(f"Approximated root: p={p:0.6f}")
    print(df)