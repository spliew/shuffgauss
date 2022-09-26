"""
Runs privacy accounting for MNIST parameters.
Outputs epsilon given inital eps0, number of update and other parameters.
"""

import numpy as np
import shuffgauss as sg
from math import exp, sqrt
from scipy.special import erf

def classic_gaussian_ldp(epsilon:float, delta:float) -> float:
    return np.sqrt(2 * np.log(1.25/delta))/(epsilon/2)

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)
    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma

if __name__ == "__main__":
    n = 6e4
    batch_size = 64
    epoch = 10
    sample_rate = batch_size/60000
    m = int(n * sample_rate)
    epsilon = 12
    delta = 1e-5
    mxlmbda = 20
    updates = int(60000*epoch/batch_size)

    sigma = calibrateAnalyticGaussianMechanism(epsilon, delta, 1)
    print(n, sample_rate, m, epsilon, delta, updates, sigma)
    ssg = sg.ApproxSCIGaussRDPtoDP(sigma, n, m, mxlmbda)
    ssg.get_subshuff()
    total_eps, lmbda = ssg.get_eps(delta, updates)
    total_eps_10, lmbda_10 = ssg.get_eps(delta, 10*updates)
    total_eps_01, lmbda_01 = ssg.get_eps(delta, 0.1*updates)
    print(f'eps: {total_eps}, lmbda:{lmbda}, updates: {updates}, sigma (2 times gs):{2*sigma}')
    print(f'eps: {total_eps_10}, lmbda:{lmbda_10}, updates: {10*updates}, sigma (2 times gs):{2*sigma}')
    print(f'eps: {total_eps_01}, lmbda:{lmbda_01}, updates: {0.1*updates}, sigma (2 times gs):{2*sigma}')
