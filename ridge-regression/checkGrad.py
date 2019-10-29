import ParallelRegression as PR
import numpy as np

def checkGrad(f, localGradient, estimateGrad):
    """
    f : arges:(x,y,beta) , return float
    estimateGrad: args:(fun,x,delta) , return grad
    localGradient: args: (x,y,beta) , return grad
    """
    delta = 1e-5
    y = 1.0
    x = np.array([np.cos(t) for t in range(5)])
    beta = np.array([np.sin(t) for t in range(5)])
    computed_grad = localGradient(x,y,beta)
    estim_grad = estimateGrad(lambda betaVar: f(x,y,betaVar) , beta, delta)
    error = np.linalg.norm(computed_grad-estim_grad, ord=1)\
            /(np.linalg.norm(computed_grad, ord=1)+np.linalg.norm(estim_grad, ord=1))
    return error < 1e-5

def checkGrad_X(F, gradient, estimateGrad, sc):
    """
    F : arges:(data,beta,lam = 0) , return float
    estimateGrad: args:(fun,x,delta) , return grad
    gradient: args: (data,beta,lam = 0) , return grad
    """
    delta = 1e-5
    data = PR.readData('data/small.test',sc)
    beta = np.array([np.sin(t) for t in range(50)])
    computed_grad = gradient(data,beta, lam=1.0)

    estim_grad = estimateGrad(lambda betaVar: F(data,betaVar, lam=1.0) , beta, delta)
    error = np.linalg.norm(computed_grad-estim_grad, ord=1)\
            /(np.linalg.norm(computed_grad, ord=1)+np.linalg.norm(estim_grad, ord=1))
    return error < 1e-5