"""Compute the exact value of kappa, a lower bound for cond(f,A,b,t).
   Author: Edvin Deadman, 2015
"""

import numpy as np
import scipy as sp
from scipy.linalg import norm, expm_frechet
from scipy.sparse.linalg import expm

def kappa(A,b,t = 1.0):
    """
    Implementation of Algorithm 3.1 from the reference below, with f = exp.

    Kappa is a lower bound for the relative condition number of exp(tA)b in the
    1-norm and is within a factor 6\sqrt(n) of the true condition number.

    Parameters
    ----------
    A: matrix stored as an n by n numpy array
    b: vector stored as a size n numpy array
    t: scalar, optional; if not supplied, the condition number of exp(A)b is
       estimated instead

    Returns
    -------
    kappa: scalar; condition estimate

    Notes
    -----

    Reference: Estimating the condition number of f(A)b, Edvin Deadman,
    Numerical Algorithms DOI: 10.1007/s11075-014-9947-4

    This code has been tested in Python 3.3.5, but should run in Python 2.x.

    """


    n = A.shape[1]

    # Compute e^{tA} and find its 1-norm
    etA = expm(t*A)
    norm1_etA = norm(etA, 1)

    # Compute the 1-norm of b and the 1-norm of tA
    norm1_b = norm(b, 1)
    norm1_tA = abs(t)*norm(A, 1)

    # Compute e^{tA}b and find its 1-norm
    etAb = etA.dot(b)
    norm1_etAb = norm(etAb, 1)

    # Compute the Kronecker matrix K(tA,b) and find its 2-norm
    K = np.zeros((n,n**2))

    for j in range(n):
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            ej = np.zeros(n)
            ej[j] = 1
            Ltemp = expm_frechet(t*A,np.outer(ei,ej), compute_expm=False)
            Lbtemp = np.dot(Ltemp,b)
            K[:,(j - 1) * n + i - 1] = Lbtemp[:]

    norm2_K = norm(K,2)

    # Return the computed value fo kappa
    return (2.0*sp.sqrt(n)*norm2_K*norm1_tA + norm1_etA*norm1_b)/norm1_etAb
