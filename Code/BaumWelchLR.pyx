#Function using the log-sum-exp trick#
def logSumExp(a):
    if np.all(np.isinf(a)):
        return np.log(0)
    else:
        b = np.max(a)
        return(b + np.log(np.sum(np.exp(a-b))))

def pForward(g, x):
    pXf = logSumExp(g[len(x)-1,:])
    return(pXf)

import cython
import numpy as np
cimport numpy as np

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:] forwardAlgLR(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:] phi, long[:] x):
    cdef double[:, :] g = np.zeros((n,m))
    cdef int i, j, l
    for i in range(0,m):
        g[0,i] = (pi[i]) + (phi[i, x[0]])
    
    
    for j in range(1, n):
        for l in range(0, m):
            g[j,l] = logSumExp(np.asarray(g[j-1, :])+np.asarray(Tmat[:,l])+(phi[l,x[j]]))
    return(g)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:] backwardAlgLR(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:] phi, long[:] x):
    cdef double[:, :] r = np.zeros((n,m))
    cdef int j, l
    for j in range(n-2, -1, -1):
        for l in range(0, m):
            r[j, l] = logSumExp(np.asarray(r[j+1,: ]) + np.asarray(Tmat[l,:]) + phi[:, x[j+1]])
    
    return(r)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple first_orderLR(int n, int m, int k, long[:] x, double tol):
    #randomly initialize pi, phi and T#
    cdef double[:] vals = np.random.rand(m)
    cdef double[:, :] vals1
    cdef double[:, :] vals2
    cdef double[:] pi = np.zeros(m)
    cdef double[:,:] Tmat = np.zeros(shape = (m, m))
    cdef double[:,:] phi = np.zeros(shape = (m, k))
    cdef double[:,:] gamma = np.zeros(shape = (n, m))
    cdef double[:,:,:] beta = np.zeros(shape = (n,m,m))
    cdef int i, t, j, w
    cdef int iterations = 0, convergence = 0, count = 0
    cdef double pOld = 1E10
    cdef double pNew = 0
    cdef double[:] indicies 
    cdef double criteria = 0
    
    vals1 = np.random.rand(m,m)
    vals2 = np.random.rand(m,k)
    Tmat = np.triu(vals1)
    Tmat = np.log(Tmat/np.sum(Tmat, axis=1)[:,None])
    phi = np.log(vals2/np.sum(vals2, axis = 1)[:,None])
    pi[0] = 1
    pi = np.log(pi)
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        #Perform forward and backward algorithms# 
        g = forwardAlgLR(n, m, k, pi, Tmat, phi, x)
        h = backwardAlgLR(n, m, k, pi, Tmat, phi, x)
        pNew = pForward(g, x)
        
        ##E-Step##
    
        #Calculate gamma and beta#
        for t in range(0, n):
            for i in range(0,m):
                gamma[t,i] = g[t,i] + h[t,i] - pNew
        
        for t in range(1, n):
            for i in range(0, m):
                for j in range(0, m):
                    if j<i:
                        beta[t,i,j] = -10E100
                
                    else:
                        beta[t,i,j] = Tmat[i,j] + phi[j, x[t]] + g[t-1, i] + h[t, j] - pNew
        ##M-Step##
    
        #Update pi, phi and Tmat#
        
#         pi = gamma[0,:] - logSumExp(gamma[0,:])
        for i in range(0, m):
            for j in range(0, m):
                Tmat[i,j] = logSumExp(beta[1::, i, j]) - logSumExp(beta[1::, i,:])
        for i in range(0,m):
            for w in range(0, k):
                j = 0
                count = 0
                for t in range(0,n):
                    if x[t] == w:
                        count = count+1
                indicies = np.zeros(count)
                for t in range(0,n):
                    if x[t] == w:
                        indicies[j] = gamma[t,i]
                        j = j+1
                    
                phi[i,w] = logSumExp(indicies) - logSumExp(gamma[:,i])
        criteria = abs(pOld - pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 1000:
            convergence = 1
        
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
    return (iterations, pNew, np.exp(pi), np.exp(phi), np.exp(Tmat))

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:,:] forwardAlg2LR(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:,:] T2mat, double[:,:] phi, long[:] x):
    cdef double[:] g = np.zeros(m)
    cdef double[:,:,:] alpha = np.zeros((n, m, m))
    cdef int t, j, l = 0
    
    g = pi + np.asarray(phi[:, x[0]])
    
    for t in range(1,n):
        for j in range(0,m):
            for l in range(0,m):
                if t ==1:
                    alpha[1,j,l] = g[j] + Tmat[j,l] + phi[l, x[1]]
                else:
                    alpha[t,j,l] = logSumExp(np.asarray(alpha[t-1,:,j]) + np.asarray(T2mat[:,j,l]) + phi[l, x[t]])
    return(alpha)

def pForward2(m,g, x):
    pXf = logSumExp(g[len(x)-1,:,m-1])
    return(pXf)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:,:] backwardAlg2LR(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:,:] T2mat, double[:,:] phi, long[:] x):
    cdef double[:,:,:] beta = np.zeros((n,m,m))
    cdef int t,j,l
    for t in range(n-2, -1, -1):
        for j in range(0, m):
            for l in range(0,m):
                beta[t,j, l] = logSumExp(np.asarray(beta[t+1,j,: ]) + np.asarray(T2mat[j,l,:]) + np.asarray(phi[:, x[t+1]]))
    
    return(beta)

#Function to return p(x_1:n) from matrix from backward algorithm
def pBackward2(m,r, pi, phi, x):
    pXb = logSumExp(r[0,:,m-1]+ pi +phi[:,x[0]])
    return(pXb)



@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple second_orderLR(int n, int m, int k, long[:] x, double[:] pi, double[:,:] Tmat, double[:,:] phi, double tol):
    #randomly initialize T2mat#
    cdef double[:,:,:] T2mat = np.zeros(shape = (m,m,m))
    cdef int i, j, t, l = 0
    cdef int iterations = 0
    cdef int convergence = 0
    cdef double pOld = 1E10
    cdef double[:, :, :] vals
    cdef double[:,:,:] alpha
    cdef double[:,:,:] beta
    cdef double pNew = 0
    cdef double[:,:,:,:] eta
    
    vals = np.random.rand(m,m,m)
    T2mat = np.triu(vals)
    T2mat = np.log(T2mat/np.sum(T2mat, axis=2)[:,:,None])
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        
        #Perform forward and backward algorithms# 
        alpha = forwardAlg2LR(n, m, k, pi, Tmat, T2mat, phi, x)
        beta = backwardAlg2LR(n, m, k, pi, Tmat, T2mat, phi, x)
        pNew = pForward2(m,alpha, x)
        ##M-Step##
        eta = np.zeros((n,m,m,m))
        #Update pi, phi and Tmat#

        for t in range(1, n-1):
            for i in range(0, m):
                for j in range(0, m):
                    for l in range(0,m):
                        eta[t,i,j,l] = alpha[t,i,j] + T2mat[i,j,l] + phi[l, x[t+1]] + beta[t+1, j, l] - pNew
        
        for i in range(0, m):
            for j in range(0, m):
                for l in range(0,m):
                        T2mat[i,j,l] = logSumExp(eta[1::,i,j,l]) - logSumExp(eta[1::,i,j,:])
        T2mat = np.exp(T2mat)
        T2mat = np.triu(T2mat)
        T2mat = np.log(T2mat/np.sum(T2mat, axis=2)[:,:,None])
        print(iterations)
        criteria = abs(pOld - pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
    return (iterations, np.exp(T2mat))
    



@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:,:, :] forwardAlg3LR(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:,:] T2mat, 
                                double[:,:,:,:] T3mat, double[:,:] phi, long[:] x):
    cdef double[:] g = np.zeros(m)
    cdef double[:, :] g2 = np.zeros((m,m))
    cdef double[:,:,:, :] alpha = np.zeros((n, m, m, m))
    cdef int t, j, q, l = 0
    
    g = pi + np.asarray(phi[:, x[0]])
    
    for t in range(1,n):
        for j in range(0,m):
            for q in range(0,m):
                for l in range(0,m):
                    if t ==1:
                        g2[j,q] = g[j] + Tmat[j,q] + phi[q, x[1]]
                        alpha[1,j,q,l] = g2[j, q] + T2mat[j,q,l] + phi[l, x[2]]
                    else:
                        alpha[t,j,q, l] = logSumExp(np.asarray(alpha[t-1,:,j, q]) + np.asarray(T3mat[:,j,q, l]) + phi[l, x[t]])
    return(alpha)

def pForward3(m,g, x):
    pXf = logSumExp(g[len(x)-1,:,m-1, m-1])
    return(pXf)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:,:, :] backwardAlg3LR(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:,:,:] T3mat, double[:,:] phi, long[:] x):
    cdef double[:,:,:, :] beta = np.zeros((n,m,m,m))
    cdef int t,j,q,l = 0
    for t in range(n-2, -1, -1):
        for j in range(0, m):
            for q in range(0,m):
                for l in range(0,m):
                    beta[t,j, k, l] = logSumExp(np.asarray(beta[t+1,j,q,: ]) + np.asarray(T3mat[j,q,l,:]) + np.asarray(phi[:, x[t+1]]))
    
    return(beta)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple third_orderLR(int n, int m, int k, long[:] x, double[:] pi, double[:,:] Tmat, double[:,:, :] T2mat, double[:,:] phi, double tol):
    #randomly initialize T3mat#
    cdef double[:,:,:, :] T3mat = np.zeros(shape = (m,m,m,m))
    cdef int i, j, q, t, l = 0
    cdef int iterations = 0
    cdef int convergence = 0
    cdef double pOld = 1E10
    cdef double[:, :, :, :] vals
    cdef double[:,:,:, :] alpha
    cdef double[:,:,:,:] beta
    cdef double pNew = 0
    cdef double[:,:,:,:,:] eta
    
    vals = np.random.rand(m,m,m,m)
    T3mat = np.triu(vals)
    T3mat = np.log(T3mat/np.sum(T3mat, axis=3)[:,:,:,None])
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        
        #Perform forward and backward algorithms# 
        alpha = forwardAlg3LR(n, m, k, pi, Tmat, T2mat, T3mat, phi, x)
        beta = backwardAlg3LR(n, m, k, pi, Tmat, T3mat, phi, x)
        pNew = pForward3(m,alpha, x)
        ##M-Step##
        eta = np.zeros((n,m,m,m, m))
        #Update pi, phi and Tmat#

        for t in range(1, n-1):
            for i in range(0, m):
                for j in range(0, m):
                    for q in range(0, m):
                        for l in range(0,m):
                            eta[t,i,j,q, l] = alpha[t,i,j, q] + T3mat[i,j,q,l] + phi[l, x[t+1]] + beta[t+1, j, q, l] - pNew
        
        for i in range(0, m):
            for j in range(0, m):
                for q in range(0, m):
                    for l in range(0,m):
                            T3mat[i,j,q,l] = logSumExp(eta[1::,i,j, q, l]) - logSumExp(eta[1::,i,j,q,:])
        T3mat = np.exp(T3mat)
        T3mat = np.triu(T3mat)
        T3mat = np.log(T3mat/np.sum(T3mat, axis=3)[:,:,:, None])
        print(iterations)
        criteria = abs(pOld - pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1

    return (iterations, np.exp(T3mat))