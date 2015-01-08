""" Examples for cond_est and cond_exact modules

Estimating the condition number of f(tA)b. In this case f = exp.

cond_est contains the function kappa_est which returns an estimate of kappa,
a lower bound for the condition number of exp(tA)b, as computed by Algorithm 4.1
in the reference below.

cond_exact contains the function kappa which returns the exact value of kappa
, a lower bound for the condition number of exp(tA)b, as computed by Algorithm
3.1 in the reference below.

Reference: Estimating the condition number of f(A)b, Edvin Deadman, Numerical
Algorithms DOI: 10.1007/s11075-014-9947-4

This code has been tested in Python 3.3.5, but should run in Python 2.x provided
the print statements are altered.

Author: Edvin Deadman, 2015

"""

import numpy as np
import scipy as sp
from scipy import linalg
from scipy.sparse import rand
from scipy.sparse.linalg import aslinearoperator

from cond_est import kappa_est
from cond_exact import kappa

np.set_printoptions(precision=3)

print( 'Demonstrating the use of routines to compute kappa, '
       'a lower bound for the condition number of f(A)b.\n' )

print( 'Small, dense matrix:\n' )

n = 10

A = np.random.rand(n,n)
b = np.ones(n)
t = 1.0

print('A = ', A)
print('b = ', b)
print('t = ', t)

print('\nWe compute a quantity kappa - a lower bound for the condition number.')
print('Estimated value of kappa for exp(tA)b: %6.4f' %kappa_est(A,b,t) )
print('Exact value of kappa for exp(tA)b: %6.4f' %kappa(A,b,t) )

print( '\nSparse matrix:' )

n=100

print( 'A can be stored in any sparse format, or as a Scipy LinearOperator.' )
A = rand(n,n,0.002,'csr')
b = np.random.rand(n)
t = 0.8

print('A = ', A)
print('b = ', b)
print('t = ', t)

print('\nEstimated value of kappa for exp(tA)b: %6.4f' %kappa_est(A,b,t) )

print('\nThe same matrix stored as a linear operator:')
Al = aslinearoperator(A)
Al.T = aslinearoperator(A.T)
# We expect a slightly different result since a different random starting vector
# has been used this time.
print('\nEstimated value of kappa for exp(tA)b: %6.4f' %kappa_est(Al,b,t) )
