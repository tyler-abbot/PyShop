"""
Origin: PyShop Session 3.
Filename: fem_course.py
Author: Tyler Abbot
Last modified: 11 october, 2015

This program is meant for the lecture on advanced numerical methods and array
manipulation, session 3 of the Python for Economists Workshop.  It is based
on the finite state example from McGratten 1998.

Some of these functions are based on those written and published by Paul
Pichler alongside his paper Solving the Multi-Country Real Business Cycle Model
Using a Monomial Rule Galerkin Method. Journal of Economic Dynamics and Control
(2011), Volume 35(2), p. 240-251.

The loop based code is based on McGratten's original Fortran implimentation.

"""
import numpy as np
import scipy.optimize
import scipy.linalg
import time


def steady(alpha, beta, delta):
    """
    A function to compute the steady state of the simple RBC model.

    Variables:
        alpha   :   float; capital share
        beta    :   float; time discount
        delta   :   float; depreciation rate

    """
    k_star = ((1 - (1 - delta)*beta)/(beta * alpha))**(1/(alpha - 1))
    return k_star, k_star**alpha - delta*k_star


def explicit_transition(states):
    """
    A function to generate an explicit transition matrix.

    Variables:
        states  :   int; number of finite states in the chain

    """
    if states == 5:
        PIE = np.array([[0.1, 0.1, 0.6, 0.1, 0.1],
                        [0.1, 0.1, 0.6, 0.1, 0.1],
                        [0.1, 0.1, 0.6, 0.1, 0.1],
                        [0.1, 0.1, 0.6, 0.1, 0.1],
                        [0.1, 0.1, 0.6, 0.1, 0.1]])
    return PIE


def scalup(points, upper, lower):
    """
    A linear transformation for state variables: [-1,1]-> [k_low, k_high].

    Variables:
        points  :   ndarray; a vector of points on [-1,1]
        upper   :   ndarray; a vector of upper bounds for the element in which
                    each point is to be projected.
        lower   :   ndarray; a vector of lower bounds for the element in which
                    each point is to be projected.

    Returns:
        ndarray cotaining the transformed points
    """
    return (points + 1.)*(upper - lower)/2 + lower


def scaldwn(points, upper, lower):
    """
    A linear transformation for state variables: [k_low, k_high] -> [-1,1].

    Variables:
        points  :   ndarray; a vector of points on [k_low, k_high]
        upper   :   ndarray; a vector of upper bounds for the element in which
                    each point is to be projected.
        lower   :   ndarray; a vector of lower bounds for the element in which
                    each point is to be projected.
    Returns:
        ndarray cotaining the transformed points

    """
    return 2*(points - lower)/(upper - lower) - 1


def policy(a_p, k_p, coeff, k, a):
    """
    A function to calculate the capital policy at points.

    Variables:
        points  :   ndarray; a vector of points on
                    [k_low, k_high]x{a1...aI}
        coeff   :   ndarray; a vector of coefficients
        k       :   ndarray; a vector containing the
                    partition of the capital state.
        a       :   ndarray; a vector containing the
                    partition of the productivity state.
    Returns:
        ndarray evaluated funciton

    """
    #Unpack the points
    #k_p = points[:, 0]
    #a_p = points[:, 1]

    #Find the elements for the points
    ll = np.array([np.where(point >= np.append([0], k[:-1]))[0][-1]
                   for point in k_p][0])
    ll[ll > 0] -= 1

    ii = np.array([np.where(point == a)[0] for point in a_p][0])
    #ii[ll > 0] -= 1

    #Define element bounds for next period
    k1p = k[ll]
    k2p = k[ll + 1]

    #Calculate the interpolators
    x = np.scaldwn(k_p, k2p, k1p)
    bs1 = 0.5*(1 - x)
    bs2 = 0.5*(1 + x)

    #Return the polciy function of the given points
    return coeff[ii, ll]*bs1 + coeff[ii, ll+1]*bs2


def mcgratten_weighted_residual(coeff):
    """
    A looped function of the McGratten problem.

    Variables:
        coeff   :   ndarray; a vector of coefficients for the parameterized
                    policy function
    Returns:
        ndarray containing the residuals.

    """
    #Unpack the arguments
    alpha, delta, I, L, k, abcissas, M, a, PIE = args
    coeff = coeff.reshape(I, L)

    #Initialize the residuals
    RES = np.zeros((I*L))

    #Construct the vector of residual funcitons
    for i in range(0, I):
        for l in range(0, L - 1):
            #Compute f at all of the quadrature points on the element
            for m in range(0, M):
                x = abcissas[m]
                kt = scalup(x, k[l+1], k[l])
                u = weights[m]
                bs1 = 0.5*(1 - x)
                bs2 = 0.5*(1 + x)

                #Compute capital next period by projection
                kp = coeff[i, l]*bs1 + coeff[i, l+1]*bs2

                #Compute consumption
                y = a[i]*kt**alpha
                c = y + (1 - delta)*kt - kp

                #Find the element for next periods capital stock
                ii = 0
                for h in range(0, L - 1):
                    if kp >= k[h]:
                        ii = h
                k1p = k[ii]
                k2p = k[ii + 1]

                #Calculate next period capital policy
                x = scaldwn(kp, k2p, k1p)
                bs1p = 0.5*(1 - x)
                bs2p = 0.5*(1 + x)

                #Initialize summation Variables
                sum1 = 0.0

                #Compute the summation and derivative simultaneously
                for h in range(0, I):
                    kpp = coeff[h, ii]*bs1p + coeff[h, ii+1]*bs2p
                    yp = a[h]*kp**alpha
                    cp = yp + (1 - delta)*kp - kpp
                    dudcp = cp**(-gamma)
                    dydkp = alpha*a[h]*kp**(alpha - 1)
                    sum1 += PIE[i, h]*dudcp*(dydkp + (1 - delta))

                ind = l + i*L

                #Generate the full residual entry by mulying qua*res*weight
                res = (beta*sum1 - c**(-gamma))*u
                RES[ind] += res*bs1
                RES[ind+1] += res*bs2

    return RES


def mcgratten_weighted_residual_vec_1(coeff):
    """
    A partially vectorized function of the McGratten problem.

    Variables:
        coeff   :   ndarray; a vector of coefficients for the parameterized
                    policy function
    Returns:
        ndarray containing the residuals.

    """
    #Unpack the arguments
    alpha, delta, I, L, k, abcissas, M, a, PIE = args
    coeff = coeff.reshape(I, L)

    #Initialize the residuals
    RES = np.zeros((I*L))

    #Construct the vector of residual funcitons
    for i in range(0, I):
        for l in range(0, L - 1):
            #Compute f at all of the quadrature points on the element
            for m in range(0, M):
                x = abcissas[m]
                kt = scalup(x, k[l+1], k[l])
                u = weights[m]
                bs1 = 0.5*(1 - x)
                bs2 = 0.5*(1 + x)

                #Compute capital next period by projection
                kp = coeff[i, l]*bs1 + coeff[i, l+1]*bs2

                #Compute consumption
                y = a[i]*kt**alpha
                c = y + (1 - delta)*kt - kp

                #Find the element for next periods capital stock
                ii = 0
                for h in range(0, L - 1):
                    if kp >= k[h]:
                        ii = h
                k1p = k[ii]
                k2p = k[ii + 1]

                #Calculate next period capital policy
                x = scaldwn(kp, k2p, k1p)
                bs1p = 0.5*(1 - x)
                bs2p = 0.5*(1 + x)

                #Initialize summation Variables
                sum1 = 0.0

                #Compute the summation and derivative simultaneously
                kpp = coeff[:, ii]*bs1p + coeff[:, ii+1]*bs2p
                yp = a*kp**alpha
                cp = yp + (1 - delta)*kp*np.ones(I) - kpp
                dudcp = cp**(-gamma)
                dydkp = alpha*a*kp**(alpha-1)
                sum1 = np.dot(PIE[i, :], dudcp*(dydkp + 1 - delta))

                ind = l + i*L

                #Generate the full residual entry by mulying qua*res*weight
                res = (beta*sum1 - c**(-gamma))*u
                RES[ind] += res*bs1
                RES[ind+1] += res*bs2

    return RES


def mcgratten_weighted_residual_vec_2(coeff):
    """
    A partially vectorized function of the McGratten problem.

    Variables:
        coeff   :   ndarray; a vector of coefficients for the parameterized
                    policy function
    Returns:
        ndarray containing the residuals.

    """
    #Unpack the arguments
    alpha, delta, I, L, k, abcissas, M, a, PIE = args
    coeff = coeff.reshape(I, L)

    #Initialize the residuals
    RES = np.zeros((I*L))

    #Construct the vector of residual funcitons
    for i in range(0, I):
        for l in range(0, L - 1):
            #Compute f at all of the quadrature points on the element
            x = abcissas
            kt = scalup(x, k[l+1], k[l])
            u = weights
            bs1 = 0.5*(1 - x)
            bs2 = 0.5*(1 + x)

            #Compute capital next period by projection
            kp = coeff[i, l]*bs1 + coeff[i, l + 1]*bs2
            #print bs1
            #Compute consumption
            y = a[i]*kt**alpha
            c = y + (1 - delta)*kt - kp

            #Find indices for next period's capital stock for each point
            ii = np.array([np.where(point >= np.append([0], k[:-1]))[0][-1]
                           for point in kp])
            ii[ii > 0] -= 1

            #Find elements for next period's capital stock
            #NOTE: Here I'm using an array as an index
            k1p = k[ii]
            k2p = k[ii + 1]

            #Calculate next period's policy
            x = scaldwn(kp, k2p, k1p)
            bs1p = 0.5*(1 - x)
            bs2p = 0.5*(1 + x)

            #Initialize the summation vector
            sum1 = np.zeros((M))

            #Compute all of the sums simultaneously
            #NOTE: You can slice AND index by an array at the same time!
            #NOTE: This is a good time to talk about broadcasting
            #The resulting matrix contains columns corresponding to each
            #abcissa, and rows corresponding to possible states in period
            #t+1
            kpp = coeff[:, ii]*bs1p + coeff[:, ii+1]*bs2p
            yp = np.outer(a, kp**alpha)
            cp = yp + (1 - delta)*np.outer(np.ones(I), kp) - kpp
            dudcp = cp**(-gamma)
            dydkp = np.outer(a, kp**(alpha - 1))*alpha
            sum1 = np.dot(PIE[i, :], dudcp*(dydkp + 1 - delta))

            #Calculate the residual functions and fill the vector
            ind = l + i*L
            res = (beta*sum1 - c**(-gamma))*u
            RES[ind] += np.dot(res, bs1)
            RES[ind + 1] += np.dot(res, bs2)
    return RES


def mcgratten_weighted_residual_vec_3(coeff):
    """
    A partially vectorized function of the McGratten problem.

    Variables:
        coeff   :   ndarray; a vector of coefficients for the parameterized
                    policy function
    Returns:
        ndarray containing the residuals.

    """
    #Unpack the arguments
    alpha, delta, I, L, k, abcissas, M, a, PIE = args
    coeff = coeff.reshape(I, L)

    #Initialize the residuals
    RES = np.zeros((I*L))

    #Construct the vector of residual funcitons
    for i in range(0, I):
        #Vectorize element loop along k axis
        #Scale and calculate capital
        x = np.kron(np.ones(L - 1), abcissas)
        kt = scalup(x, np.kron(k[1:], np.ones(M)), np.kron(k[:-1], np.ones(M)))
        u = np.kron(np.ones(L - 1), weights)
        bs1 = 0.5*(1 - x)
        bs2 = 0.5*(1 + x)
        Bs1 = 0.5*(1 - abcissas)
        Bs2 = 0.5*(1 + abcissas)

        #Compute capital next period by projection
        kp = np.kron(coeff[i, :-1], np.ones(M))*bs1\
            + np.kron(coeff[i, 1:], np.ones(M))*bs2

        #Compute consumption
        y = a[i]*kt**alpha
        c = y + (1 - delta)*kt - kp

        #Find indices for next period's element bounds
        ii = np.array([np.where(point >= np.append([0], k[:-1]))[0][-1]
                       for point in kp])
        ii[ii > 0] -= 1

        #Define element bounds for next period
        k1p = k[ii]
        k2p = k[ii + 1]

        #Calculate policy next period
        x = scaldwn(kp, k2p, k1p)
        bs1p = 0.5*(1 - x)
        bs2p = 0.5*(1 + x)

        #Initialize the summation
        sum1 = np.zeros(((L-1)*M))

        #Compute all of the sums simultaneously using linear algebra
        kpp = coeff[:, ii]*bs1p + coeff[:, ii+1]*bs2p
        yp = np.outer(a, kp**alpha)
        cp = yp + (1 - delta)*np.outer(np.ones(I), kp) - kpp
        dudcp = cp**(-gamma)
        dydkp = np.outer(a, kp**(alpha - 1))*alpha
        sum1 = np.dot(PIE[i, :], dudcp*(dydkp + 1 - delta))

        #Calculate the residual functions
        res = (beta*sum1 - c**(-gamma))*u
        res = res.reshape(L-1, M)
        RES[i*L: (i+1)*L-1] += np.dot(res, Bs1)
        RES[i*L + 1: (i+1)*L] += np.dot(res, Bs2)
    return RES


def mcgratten_weighted_residual_vec(coeff):
    """
    A vectorized function of the McGratten problem.

    Variables:
        coeff   :   ndarray; a vector of coefficients for the parameterized
                    policy function
    Returns:
        ndarray containing the residuals.

    """
    #Unpack the arguments
    alpha, delta, I, L, k, abcissas, M, a, PIE = args

    #Reshape coefficients in case they are vector
    coeff = coeff.reshape(I, L)

    #Initialize the residuals
    RES = np.zeros((I*L))

    #For readability define the number of rows
    H = I*(L - 1)

    #Construct the vector of residual funcitons
    #Vectorize all loops
    #Scale and calculate capital
    x = np.kron(np.ones(H), abcissas)
    kt = scalup(x, np.kron(k[1:], np.ones(I*M)), np.kron(k[:-1], np.ones(I*M)))
    u = np.kron(np.ones(H), weights)
    bs1 = 0.5*(1 - x)
    bs2 = 0.5*(1 + x)
    #NOTE: bs is constant across elements, so simplify calculations later
    Bs1 = 0.5*(1 - abcissas)
    Bs2 = 0.5*(1 + abcissas)

    #Compute capital next period by projection
    kp = np.kron(coeff[:, :-1].reshape(H,), np.ones((M)))*bs1\
        + np.kron(coeff[:, 1:].reshape(H,), np.ones((M,)))*bs2

    #Compute consumption
    y = np.kron(a, np.ones((L - 1)*M))*kt**alpha
    c = y + (1 - delta)*kt - kp

    #Find indices for next periods element bounds
    ii = np.array([np.where(point >= np.append([0], k[:-1]))[0][-1]
                   for point in kp])
    ii[ii > 0] -= 1

    #Define element bounds for next period
    k1p = k[ii]
    k2p = k[ii + 1]

    #Calculate policy next period
    x = scaldwn(kp, k2p, k1p)
    bs1p = 0.5*(1 - x)
    bs2p = 0.5*(1 + x)

    #Initialize the summation
    sum1 = np.zeros((H*M))

    #Compute the sums
    #Calculate next period's policy, output, and consumption
    kpp = coeff[:, ii]*bs1p + coeff[:, ii+1]*bs2p
    yp = np.outer(a, kp**alpha)
    cp = yp + (1 - delta)*np.outer(np.ones(I), kp) - kpp

    #Calculate marginal product of capital and marginal utility
    dudcp = cp**(-gamma)
    dydkp = np.outer(a, kp**(alpha - 1))*alpha
    rhs = dudcp*(dydkp + 1 - delta)

    #Generate a block diagonal matrix for the expectation
    mats = [rhs[:, i*M*(L - 1): (i+1)*M*(L - 1)] for i in range(0, I)]
    temp1 = scipy.linalg.block_diag(*mats)
    temp2 = PIE.reshape(I*I)
    sum1 = np.dot(temp1.T, temp2)

    #Calculate the residual functions
    res = (beta*sum1 - c**(-gamma))*u
    res = res.reshape((H, M))
    RES[[i for i in range(0, I*L) if i not in
        range((L - 1), I*L, L)]] += np.dot(res, Bs1)
    RES[[i for i in range(0, I*L) if i not in
        range(0, I*L - 1, L)]] += np.dot(res, Bs2)

    return RES

# Step 1: Calibrate Parameters
gamma = 2.0
beta = 0.98
alpha = 0.3
delta = 0.9


# Step 2: Compute steady state
k_star, c_star = steady(alpha, beta, delta)

# Step 3: Generate the state space
#Define upper and lower bound of the region of interest around the steady stat
kmin = 0.5*k_star
kmax = 1.5*k_star

#Define the finite states for the markov chain
I = 5
PIE = explicit_transition(I)
a = np.array([0.95, 0.975, 1.0, 1.025, 1.05])

#Generate a state space vector over k
L = 11
k = np.linspace(kmin, kmax, num=L)

# Step 4: Calculate quadrature weights and abcissas
#Specify number of points in the integral on each element [k_j, k_j+1]
M = 10

#Generate absissas and weights
abcissas, weights = np.polynomial.legendre.leggauss(M)

# Step 5: Guess and solve
#Generate intial guess as half of production at all nodes
coeff0 = np.outer(a, k**alpha)*0.1

#Pack up all of the arguments to pass to the solver
args = (alpha, delta, I, L, k, abcissas, M, a, PIE)

time0 = time.time()
res = scipy.optimize.newton_krylov(mcgratten_weighted_residual, coeff0,
                                   method='lgmres', verbose=True, maxiter=1000,
                                   line_search='armijo')
print(time.time() - time0)

time0 = time.time()
res1 = scipy.optimize.newton_krylov(mcgratten_weighted_residual_vec_1, coeff0,
                                    method='lgmres', verbose=True,
                                    maxiter=1000, line_search='armijo')
print(time.time() - time0)
print(np.linalg.norm(res - res1))

time0 = time.time()
res2 = scipy.optimize.newton_krylov(mcgratten_weighted_residual_vec_2, coeff0,
                                    method='lgmres', verbose=True,
                                    maxiter=1000, line_search='armijo')
print(time.time() - time0)
print(np.linalg.norm(res - res2))

time0 = time.time()
res3 = scipy.optimize.newton_krylov(mcgratten_weighted_residual_vec_3, coeff0,
                                    method='lgmres', verbose=True,
                                    maxiter=1000, line_search='armijo')
print(time.time() - time0)
print(np.linalg.norm(res - res3))

time0 = time.time()
res_vec = scipy.optimize.newton_krylov(mcgratten_weighted_residual_vec, coeff0,
                                       method='lgmres', verbose=True,
                                       maxiter=1000, line_search='armijo')
print(time.time() - time0)
print(np.linalg.norm(res - res_vec))
