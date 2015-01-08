"""Functions for computing an estimate of kappa, a lower bound for
   cond(f,A,b,t).
   Author: Edvin Deadman, 2015
"""
import numpy as np
import scipy as sp

from scipy.linalg import norm
from numpy import dot, zeros, inf, conj
from scipy.sparse.linalg import onenormest, aslinearoperator, LinearOperator

def kappa_est(Ain, b, t = 1.0, itmax = 50, trace = 0.0):
    """
    Implementation of Algorithm 4.1 from the reference below.

    Kappa is a lower bound for the relative condition number of exp(tA)b in the
    1-norm and is within a factor 6\sqrt(n) of the true condition number. This
    function computes an estimate of kappa based on a power iteration.

    Parameters
    ----------
    A:      matrix stored as an n by n numpy array, a sparse matrix, or a
            transposable linear operator
    b:      vector stored as a size n numpy array
    t:      scalar, optional; if not supplied, the condition number of exp(A)b
            is estimated instead
    itmax:  (optional) maximum number of power iterations allowed. Generally
            only 2-4 iterations are required for convergence so a value of 5-10
            is typically fine for itmax.
    trace:  (optional) the trace of matrix A. If A is supplied as a numpy array
            then this argument should be omitted since the routine will compute
            tr(A). If A is supplied as a sparse matrix or linear operator then
            the trace, when supplied, can help reduce the number of matrix-
            vector multiplications required to compute kappa. If the trace is
            not known then this argument can be omitted with no loss of
            accuracy.

    Returns
    -------
    kappa: scalar; condition estimate

    Notes
    -----

    Reference: Estimating the condition number of f(A)b, Edvin Deadman,
    Numerical Algorithms DOI: 10.1007/s11075-014-9947-4

    This code has been tested in Python 3.3.5, but should run in Python 2.x.

    """

    # A few preliminary definitions and setup
    n = Ain.shape[0]
    D = None
    Dinv = None
    Db = None
    Dinvb = None
    tol = 2**(-11) # Unit roundoff for half-precision
    balance = True
    sigma = norm(b, 1)

    # This 'if' statement executes the correct balancing and shifting code and
    # computes the parameters m and s depending on whether A is a numpy array
    # or a sparse matrix/linear operator
    if isinstance(Ain,np.ndarray):
        # A is a numpy array
        alpha = norm(Ain,'fro')
        # Call LAPACK balancing routine since we explicitly have access to A
        pivscale = sp.linalg.lapack.dgebal(Ain,scale = True)[3]
        D = np.diag(pivscale)
        Dinv = np.diag(1/pivscale)
        Atilde = dot(Ain,D)
        Atilde = dot(Dinv,Atilde)
        # Check if dgebal reduced the norm of A
        norm1_A = norm(Ain,1)
        norm1_Ain = norm1_A
        norm1_Atilde = norm(Atilde,1)

        if norm1_Atilde < norm1_A:
            # If the norm was reduced, use the balanced matrix
            A = Atilde
            b = dot(Dinv,b)
            norm1_A = norm1_Atilde
        else:
            # If the norm was not reduced, use the original matrix
            A = Ain
            balance = False

        # Store the balancing matrix D and its inverse Dinv and their
        # conjugates as linear operators
        Db = conj(D)
        Dinvb = conj(Dinv)
        D = aslinearoperator(D)
        Dinv = aslinearoperator(Dinv)
        Db = aslinearoperator(Db)
        Dinvb = aslinearoperator(Dinvb)

        # Shift A by its trace/n
        mu = A.trace() / n
        A = A - mu*np.eye(n,n)

        # Compute the truncation and scaling parameters m and s
        m, s = get_m_and_s(t*A)

    else:
        # A is a sparse matrix or a linear operator
        norm1_A = onenormest(Ain,t=2)
        norm1_Ain = norm1_A

        # Call the sparse matrix balancing algorithm of Chen and Demmel
        Atilde, D, Dinv, Db, Dinvb = krylov_cutoff(Ain, norm1_A)
        norm1_Atilde = onenormest(Atilde,t=2)

        # Check if balancing reduced the norm of A
        if norm1_Atilde < norm1_A:
            # If the norm was reduced, use the balanced matrix
            b = Dinv.dot(b)
            norm1_A = norm1_Atilde
            A = Atilde
            A.T = Atilde.T
        else:
            # If the norm was not reduced, use the original matrix
            A = Ain
            balance = False

        if trace == 0.0:
            # No trace was supplied so use set mu = 0 (no shifting)
            mu = 0
        else:
            # Shift the linear operator so we use A-mu*I rather than A
            mu = trace / n
            A = ShiftedLinearOperator(A, mu)

        # Compute the truncation and scaling parameters m and s
        Ascale = t*A
        Ascale.T = t*A.T
        m, s = get_m_and_s(Ascale)

    eta = np.exp(t * mu / s)
    s = int(s)

    # Choose a unit non-zero starting vector, to begin the power iteration
    y_k = np.random.rand(n)
    y_k = y_k / norm(y_k, 2)
    gamma_k = 1.0

    for k in range(itmax):
        # Power iteration to estimate norm of Kronecker matrix
        if balance:
             y_k = Db.dot(y_k)

        F1 = zeros(n)
        F2 = b
        b1 = zeros(n)
        b2 = b

        # This next loop computes the iterative term y_{k+1}
        for i in range(s):
            c1 = max(norm(b1, inf), norm(b2, inf))

            for j in range(1, m + 1):
                G1 = zeros(n)
                G2 = b2
                d1 = zeros(n)
                d2 = b2

                if balance:
                    d2 = Db.dot(D.dot(d2))

                for q in range(s):
                    c3 = max(norm(d1, inf), norm(d2, inf))

                    for r in range(1, m + 1):
                        d1 = (conj(t * A.T.dot(conj(d1))) + \
                              y_k * dot(conj(b), d2)) / (s * r)
                        d2 = conj(t * A.T.dot(conj(d2))) / (s * r)
                        c4 = max(norm(d1, inf), norm(d2, inf))
                        G1 = G1 + d1
                        G2 = G2 + d2

                        if c3 + c4 <= tol * max(norm(G1, inf), norm(G2, inf)):
                            break

                        c3 = c4

                    G1 = conj(eta) * G1
                    G2 = conj(eta) * G2
                    d1 = G1
                    d2 = G2

                if balance:
                    G1 = Dinv.dot(Dinvb.dot(G1))

                b1 = (t * A.dot(b1) + G1) / (s * j)
                b2 = (t * A.dot(b2)) / (s * j)
                c2 = max(norm(b1, inf), norm(b2, inf))
                F1 = F1 + b1
                F2 = F2 + b2

                if c1 + c2 <= tol * max(norm(F1, inf), norm(F2, inf)):
                    break

                c1 = c2

            F1 = eta * F1
            F2 = eta * F2
            b1 = F1
            b2 = F2

        y_kp1 = F1

        if balance:
            y_kp1 = D.dot(y_kp1)

        iterations = k + 1

        # Computation of y_{k+1} complete; now test for convergence
        gamma_kp1 = np.sqrt(norm(y_kp1, 2))
        if abs(gamma_kp1 - gamma_k) / gamma_kp1 < 0.1 or k == itmax - 1:
            gamma = gamma_kp1
            break

        y_k = y_kp1 / norm(y_kp1, 2)
        gamma_k = gamma_kp1

    # Estimate ||e^{tA}||_1; to do this we create a new linear operator, actexp,
    # which acts on a vector x to give e^{tA}x.
    actexp = MatrixExponentialOperator(A, s, m, eta, t, D, Db, Dinv, Dinvb,
                                       balance = balance)
    actexp.T = MatrixExponentialOperator(A, s, m, eta, t, \
                                             D, Db, Dinv, Dinvb, herm = True,
                                             balance = balance)
    # Use the one norm estimation method for this new linear operator
    beta = onenormest(actexp)


    # Now compute e^{tA}b
    F2 = b
    b1 = b
    for i in range(s):
        c1 = norm(b1, inf)
        for j in range(1, m + 1):
            b1 = t * A.dot(b1) / (s * j)
            c2 = norm(b1, inf)
            F2 = F2 + b1

            if c1 + c2 <= tol * norm(F2, inf):
                break

            c1 = c2

        F2 = eta * F2
        b1 = F2

    if balance:
        F2 = D.dot(F2)

    # Form and return the quantity kappa, our condition estimate
    return (2.0*sp.sqrt(n)*gamma * abs(t) * norm1_Ain + beta * sigma) /\
             norm(F2, 1)

################################################################################

""" Utility functions used by kappa """

def krylov_cutoff(A, normA):
    """
    This function implements the sparse matrix balancing algorithm of Chen and
    Demmel (Balancing sparse matrices for computing eigenvalues. Linear Algebra
    Appl. 309, 261–287 (2000)).

    A is a sparse matrix or linear operator, normA is its (estimated) 1-norm

    The output is a balanced matrix, Atilde and a diagonal transformation matrix
    D, together with its inverse Dinv and the associated conjugates Db and Dinvb

    All outputs are provided using the classes BalancedLinearOperator or
    DiagLinearOperator, derived from SciPy's LinearOperator class.

    """

    # Set up some useful parameters
    n = A.shape[0]
    it_max = 5
    cutoff = 1e-8
    diag = np.ones(n)

    # Begin the balancing iteration
    for i in range(it_max):

        # Form a vector of random 1s and -1s
        z = np.random.rand(n)
        for j in range(n):
            if z[j] < 0.5:
                z[j] = -1.0
            else:
                z[j] = 1.0

        p = (A.dot(z / diag)) * diag
        r = (A.T.dot(z * diag)) / diag

        for j in range(n):
            if (abs(p[j]) < normA * cutoff) or (r[j] == 0.0):
                diag[j] = diag[j]
            else:
                diag[j] = diag[j] * sp.sqrt(abs(r[j] / p[j]))

    diag = 1.0 / diag

    # Form the balanced matrix and the transformation matrices
    D = DiagLinearOperator(diag)
    Dinv = DiagLinearOperator(1 / diag)
    Atilde = BalancedLinearOperator(A, diag)
    Atilde.T = BalancedLinearOperator(A.T, 1 / diag)
    Db = DiagLinearOperator(sp.conj(diag))
    Dinvb = DiagLinearOperator(sp.conj(1 / diag))

    return Atilde, D, Dinv, Db, Dinvb


def onenormest_Ap(A,p, t = 2):
    """
    Estimate the 1-norm of A^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.

    """
    # Form a LinearOperatorPower object (derived from SciPy's LinearOperator but
    # returns A^p x rather than Ax) and call onenormest with this object.
    return onenormest(LinearOperatorPower(A,p), t = 2)


def alpha(A,p):
    # Compute the quantity alpha as used in Algorithm 4.1
    dp = onenormest_Ap(A,p) ** (1.0/p)
    dpp1 = onenormest_Ap(A,p+1) ** (1.0 / (p+1))

    return max(dp,dpp1)


def get_m_and_s(A, prec = 'half'):
    """
    Compute the truncation and scaling parameters, m and s.

    This is an implementation of the method described in Al-Mohy,A.H.
    Higham,N.J.: Computing the action of the matrix exponential, with an
    application to exponential integrators. SIAM J. Sci. Comput. 33(2),
    488–511 (2011)

    The default for our purposes is to return parameters for half precision.

    """
    from theta_values import get_theta
    theta = get_theta(prec)

    tol = roundoff[prec]
    m_max = 55
    p_max = 8
    l = 2

    # Compute the RHS of Al-Mohy and Higham eq. (3.13)
    rhs = 2 * l * theta[m_max] * p_max * (p_max + 3) / float(m_max)
    norm1_A = onenormest(A, t = 2)

    m = None
    s = None

    if norm1_A <= rhs:
        for m_temp, theta_temp in theta.items():
            s_temp = int(np.ceil(norm1_A / theta_temp))
            if m is None or m_temp * s_temp < m * s:
                m = m_temp
                s = s_temp
    else:
        for p in range(2, p_max + 1):
            temp = alpha(A, p)
            alph = temp
            for m_temp in range(p * (p - 1) - 1, m_max + 1):
                s_temp = int(np.ceil(alph / theta[m_temp]))
                if m is None or m_temp * s_temp < m * s:
                    m = m_temp
                    s = s_temp

    return m, s

roundoff = {
    'half': 2**(-11),
    'single': 2**(-24),
    'double': 2**(-53),
}

################################################################################
""" Classes derived from SciPy's LinearOperator class used to compute kappa """


class ShiftedLinearOperator(LinearOperator):
    """
    Given a matrix/operator A and a scalar mu, this operator acts as A - mu*I

    """

    def __init__(self,A,mu):
        self._A = A
        self._mu = mu
        self.shape = A.shape

    def matvec(self, x):
        x = self._A.dot(x) - self._mu * x
        return x

    def rmatvec(self, x):
        x = x.dot(self._A) - self._mu * x
        return x

    def matmat(self, X):
        X = self._A.dot(X) - self._mu * X
        return X

    @property
    def T(self):
        return ShiftedLinearOperator(self._A.T, self._mu)


class LinearOperatorPower(LinearOperator):
    """
    Given a matrix/operator A and positive integer p, this operator acts as A^p

    """
    def __init__(self,A,p):
        self._A = A
        self._p = p
        self.shape = A.shape

    def matvec(self, x):
        for i in range(self._p):
            x = self._A.dot(x)
        return x

    def rmatvec(self, x):
        for i in range(self._p):
            x = x.dot(self._A)
        return x

    def matmat(self, X):
        for i in range(self._p):
            X = self._A.dot(X)
        return X

    @property
    def T(self):
        return LinearOperatorPower(self._A.T, self._p)


class DiagLinearOperator(LinearOperator):
    """
    Given a vector, diag, this class is a linear operator acting as the matrix
    with diag on its diagonal.

    """
    def __init__(self, diag):
        self._D = diag
        self.shape = (diag.shape[0], diag.shape[0])

    def matvec(self, x):
        x = self._D * x
        return x

    def rmatvec(self, x):
        x = self._D * x
        return x

    def matmat(self, X):
        for i in range(X.shape[1]):
            X[:,i] = self._D*X[:,i]
        return X

class BalancedLinearOperator(LinearOperator):
    """
    Given a LinearOperator A and a vector diag, this new operator takes the form
    D^{-1}AD where D is the matrix with diag on its diagonal.

    """

    def __init__(self, A, diag):
        self._diag = diag
        self._A = A
        self.shape = A.shape

    def matvec(self, x):
        x = self._A.dot(self._diag * x) / self._diag
        return x

    def rmatvec(self, x):
        x = self._A.T.dot(x / self._diag) * self._diag
        return x

    def matmat(self, X):
        for i in range(X.shape[1]):
            X[:,i] =  self._A.dot(self._diag * X[:,i]) / self._diag
        return X



class MatrixExponentialOperator(LinearOperator):
    """
    Given a LinearOperator A this new operator takes the form e^{tA} so when
    acting on a vector x, e^{tA}x is returned.

    Initializing an instance of this class requires:

    A   :   matrix or operator
    s, m:   scaling and truncation parameters used in Al-Mohy and Higham's alg.
    eta :   quantity required by Al Mohy and Higham's alg.
    t   :   scalar premultiplying A
    D   :   diagonal balancing transformation (matrix or operator) - Dinv is its
            inverse and Db and Dinvb the conjugates - not used if balance=False
    herm:   should be set to True if we want e^{tA^*}, the conjugate transpose

    """

    def __init__(self, A, s, m, eta, t, D = None, Db = None, \
                 Dinv = None, Dinvb = None, herm = False, balance = False):
        self.A = A
        self.s = s
        self.m = m
        self.eta = eta
        self.t = t
        self.tol = 2**(-11)
        self.shape = A.shape
        self.herm = herm # Denotes that we want the conjugate transpose
        self.balance = balance # True if balancing is to be used
        self.D = D
        self.Dinv = Dinv
        self.Db = Db
        self.Dinvb = Dinvb

    def matvec(self, b):
        # Code to compute e^{tA}b
        if self.balance:
            if self.herm:
                b = self.Db.dot(b)
            else:
                b = self.Dinv.dot(b)

        F = b
        for i in range(self.s):
            c1 = norm(b, inf)

            for j in range(1, self.m + 1):
                if self.herm:
                    b = conj(self.t * self.A.T.dot(conj(b))) / (self.s * j)
                else:
                    b = self.t * self.A.dot(b) / (self.s * j)

                c2 = norm(b, inf)
                F = F + b

                if c1 + c2 <= self.tol * norm(F, inf):
                    break

                c1 = c2

            if self.herm:
                F = conj(self.eta)*F
            else:
                F = self.eta * F
            b = F

        if self.balance:
            if self.herm:
                F = self.Dinvb.dot(F)
            else:
                F = self.D.dot(F)

        return F

    def matmat(self, B):
        # Code to compute e^{tA}B
        if self.balance:
            if self.herm:
                B = self.Db.dot(B)
            else:
                B = self.Dinv.dot(B)

        F = B
        for i in range(self.s):
            c1 = norm(B, inf)

            for j in range(1, self.m + 1):
                if self.herm:
                    B = conj(self.t * self.A.T.dot(conj(B))) / (self.s * j)
                else:
                    B = self.t * self.A.dot(B) / (self.s * j)
                c2 = norm(B, inf)
                F = F + B

                if c1 + c2 <= self.tol * norm(F, inf):
                    break

                c1 = c2

            if self.herm:
                F = conj(self.eta) * F
            else:
                F = self.eta * F

            B = F

        if self.balance:
            if self.herm:
                F = self.Dinvb.dot(F)
            else:
                F = self.D.dot(F)

        return F
