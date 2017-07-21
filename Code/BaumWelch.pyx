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
def indicator(a, b):
    if a == b:
        return 1
    else:
        return 0
def pForwardARHMM(g, x):
    pXf = logSumExp(g[len(x)-1,:, :])
    return(pXf)

    
import cython
import numpy as np
cimport numpy as np

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:] forwardAlg(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:] phi, long[:] x):
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
cpdef double[:,:] backwardAlg(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:] phi, long[:] x):
    cdef double[:, :] r = np.zeros((n,m))
    cdef int j, l
    for j in range(n-2, -1, -1):
        for l in range(0, m):
            r[j, l] = logSumExp(np.asarray(r[j+1,: ]) + np.asarray(Tmat[l,:]) + phi[:, x[j+1]])
    
    return(r)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:] Viterbi(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:] phi, long[:] x):
    cdef double[:, :] f = np.zeros(shape = (n,m))
    cdef double[:, :] alpha = np.zeros(shape = (n,m))
    cdef double[:] zStar = np.zeros(n)
    cdef double[:] u 
    cdef int t, i
    
    for t in range(0, n):
        for i in range(0,m):
            if t == 0:
                f[0, i] = pi[i] + phi[i, x[0]]
            else:
                u = np.asarray(f[t-1, :]) + np.asarray(Tmat[:, i]) + phi[i, x[t]]
                f[t,i] = np.max(u)
                alpha[t,i] = np.argmax(u)
    zStar[n-1] = np.argmax(np.asarray(f[n-1, :]))
    for i in range(n-2, -1, -1):
        zStar[i] = alpha[i+1, int(zStar[i+1])]
    return zStar

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple first_order(int n, int m, int k, long[:] x, double tol):
    #randomly initialize pi, phi and T#
    cdef double[:] vals = np.random.rand(m)
    cdef double[:, :] vals1
    cdef double[:, :] vals2
    cdef double[:] pi = np.log(vals/np.sum(vals))
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
    #cdef double[:,:] p = np.zeros(shape = (n,m))
    
    vals1 = np.random.rand(m,m)
    vals2 = np.random.rand(m,k)
    Tmat = np.log(vals1/np.sum(vals1, axis=1)[:,None])
    phi = np.log(vals2/np.sum(vals2, axis = 1)[:,None])
    
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        #Perform forward and backward algorithms# 
        g = forwardAlg(n, m, k, pi, Tmat, phi, x)
        h = backwardAlg(n, m, k, pi, Tmat, phi, x)
        pNew = pForward(g, x)
        
        ##E-Step##
    
        #Calculate gamma and beta#
        for t in range(0, n):
            for i in range(0,m):
                gamma[t,i] = g[t,i] + h[t,i] - pNew
        #p = np.full((n,m), pNew)
        #gamma = g+h-p
        for t in range(1, n):
            for i in range(0, m):
                for j in range(0, m):
                    beta[t,i,j] = Tmat[i,j] + phi[j, x[t]] + g[t-1, i] + h[t, j] - pNew
        ##M-Step##
    
        #Update pi, phi and Tmat#
        pi = gamma[0,:] - logSumExp(gamma[0,:])
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
            print(iterations)
    return (iterations, pNew, np.exp(pi), np.exp(phi), np.exp(Tmat))
    
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:,:] forwardAlg2(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:,:] T2mat, double[:,:] phi, long[:] x):
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
cpdef double[:,:,:] backwardAlg2(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:,:] T2mat, double[:,:] phi, long[:] x):
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
cpdef tuple second_order(int n, int m, int k, long[:] x, double[:] pi, double[:,:] Tmat, double[:,:] phi, double tol):
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
    T2mat = np.log(vals/np.sum(vals, axis=2)[:,:,None])
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        
        #Perform forward and backward algorithms# 
        alpha = forwardAlg2(n, m, k, pi, Tmat, T2mat, phi, x)
        beta = backwardAlg2(n, m, k, pi, Tmat, T2mat, phi, x)
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
cpdef two_hidden_states(int n, int M, int N, int k, long[:] x, double tol):

    cdef double[:, :] valsA = np.random.rand(N,N)
    cdef double[:, :] valsphi = np.random.rand(N*M,k)
    cdef double[:,:,:] valsB = np.random.rand(N,M,M)
    cdef double[:] valspi = np.random.rand(N*M)
    cdef double[:] pi = np.log(valspi/np.sum(valspi))
    cdef double[:,:,:] B = np.zeros(shape = (N,M,M))
    cdef double[:,:] A = np.zeros(shape = (N, N))
    cdef double[:,:] phi = np.zeros(shape = (N*M, k))
    
    cdef double[:,:] gamma = np.zeros(shape = (n, N*M))
    cdef double[:,:,:] beta = np.zeros(shape = (n,N*M,N*M))
    cdef double[:,:] Tmat = np.zeros((N*M, N*M))
    
    cdef int i, t, j, w, q, l = 0
    cdef int iterations = 0, convergence = 0, count = 0, count_ik = 0, count_jl = 0
    cdef double pOld = 1E10
    cdef double pNew = 0
    cdef double[:] indicies 
    cdef double criteria = 0
    
    A = np.log(valsA/np.sum(valsA, axis=1)[:,None])
    phi = np.log(valsphi/np.sum(valsphi, axis = 1)[:,None])
    B = np.log(valsB/np.sum(valsB, axis=2)[:,:,None])
    
   
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        
        #Find Tmat which is AxB
        Tmat = np.zeros((N*M, N*M))
        count_ik = 0
        count_jl = 0
        count = 0
        for i in range(0,N):
            for q in range(0,M):
                count_jl = 0
                for j in range(0, N):
                    for l in range(0,M):
                        Tmat[count_ik,count_jl] = A[i,j] + B[j,q,l]
                        count_jl +=1
                count_ik +=1
                
        #Perform forward and backward algorithms# 
        g = forwardAlg(n, N*M, k, pi, Tmat, phi, x)
        h = backwardAlg(n, N*M, k, pi, Tmat, phi, x)
        pNew = pForward(g, x)
        
        ##E-Step##
    #Calculate gamma and beta#
        for t in range(0, n):
            for i in range(0,N*M):
                gamma[t,i] = g[t,i] + h[t,i] - pNew
        #gamma = g+h-pNew*np.ones(shape = (n, N*M))
        for t in range(1, n):
            for i in range(0, N*M):
                for j in range(0, N*M):
                    beta[t,i,j] = Tmat[i,j] + phi[j, x[t]] + g[t-1, i] + h[t, j] - pNew
        ##M-Step##
    
        #Update pi, phi and Tmat#
        pi = gamma[0,:] - logSumExp(gamma[0,:])
        
        
        
        for i in range(0,N):
            for j in range(0,N):
                Asums = []
                for q in range(i*M, (i+1)*M):
                    for l in range(j*M, (j+1)*M):
                        Asums.append(logSumExp(beta[1:n,q,l]))

                A[i,j] = logSumExp(Asums) 
        
        A = np.log(np.exp(A)/np.sum(np.exp(A), axis = 1)[:, np.newaxis])
        
        for j in range(0,N):
            for q in range(0,M):
                for l in range(0,M):
                    Bsums = []
                    for i in range(q,N*M,M):
                        Bsums.append(logSumExp(beta[1:n,i,M*j+l]))
                    B[j,q,l] = logSumExp(Bsums) 

        B = np.log(np.exp(B)/np.sum(np.exp(B), axis = 2)[:,:,np.newaxis])
        
        for i in range(0,N*M):
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
            print(iterations)
    return (iterations, pNew, np.exp(pi), np.exp(phi), np.exp(Tmat), np.exp(A), np.exp(B))
 

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef double[:,:,:, :] forwardAlg3(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:,:] T2mat, 
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
cpdef double[:,:,:, :] backwardAlg3(int n, int m, int k, double[:] pi, double[:,:] Tmat, double[:,:,:,:] T3mat, double[:,:] phi, long[:] x):
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
cpdef tuple third_order(int n, int m, int k, long[:] x, double[:] pi, double[:,:] Tmat, double[:,:, :] T2mat, double[:,:] phi, double tol):
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
    T3mat = np.log(vals/np.sum(vals, axis=3)[:,:,:,None])
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        
        #Perform forward and backward algorithms# 
        alpha = forwardAlg3(n, m, k, pi, Tmat, T2mat, T3mat, phi, x)
        beta = backwardAlg3(n, m, k, pi, Tmat, T3mat, phi, x)
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


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple HSMM(int n, int m, int k, long[:] x, double tol):
    cdef double[:] pi = np.zeros(m)
    cdef double[:,:] Tmat = np.zeros(shape = (m, m))
    cdef double[:,:] phi = np.zeros(shape = (m, k))
    cdef double it=0, p = 0
    cdef int iterations = 0, convergence = 0, count = 0
    cdef double criteria = 0
    cdef double[:, :, :] eta = np.zeros(shape=(n, m, m))
    cdef double[:,:] gamma = np.zeros(shape = (n, m))
    cdef double[:] zStarOld = (n+1)*np.ones(n)
    cdef double[:] zStar = np.zeros(n)
    cdef int t, i, j, w
    cdef double[:] indicies 
    # First run 1st order Baum-Welch for estimate of parameters
    it, p, pi, phi, Tmat = first_order(n, m, k, x, 10)
    
    #Until convergence (zStar between iterations differs in less than tol locations):
    while convergence == 0:
        #Perform Viterbi Algorithm# 
        zStar = Viterbi(n,  m,  k,  pi, Tmat, phi,  x)
        
        for t in range(0, n):
            for i in range(0, m):
                gamma[t, i] = indicator(zStar[t], i)
                for j in range(0, m):
                    if t != n-1:
                        eta[t, i, j] = indicator(zStar[t], i)*indicator(zStar[t+1], j)
                    
        ##M-Step##
    
        #Update pi, phi and Tmat#
        pi = np.log(np.asarray(gamma[0,:])/np.sum(np.asarray(gamma[0,:])))
        for i in range(0, m):
            for j in range(0, m):
                Tmat[i,j] = np.log(np.sum(np.asarray(eta[:-1, i, j]))/np.sum(np.asarray(eta[:-1, i,:])))
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
                    
                phi[i,w] = np.log(np.sum(indicies)/np.sum(np.asarray(gamma[:,i])))
        
       
        for i in range(n):
            if zStarOld[i] != zStar[i]:
                count += 1
        criteria = count
        count = 0
        if criteria < tol:
            convergence = 1
        elif iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            zStarOld = zStar
            iterations +=1
            print(iterations)
    return (iterations, criteria, np.exp(pi), np.exp(phi), np.exp(Tmat))




@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.double_t, ndim=3] forwardAlgARHMM(int n, int m, int k, np.ndarray[np.double_t, ndim=1] pi, 
                                                      np.ndarray[np.double_t, ndim=2] Tmat, 
                                                      np.ndarray[np.double_t, ndim=2] phi, 
                                                      np.ndarray[np.double_t, ndim=3] psi, 
                                                      long[:] x):
    cdef np.ndarray[np.double_t, ndim=3] g
    g = np.zeros((n,m, k))
    cdef int i, j,t
    for i in range(0,m):
        g[0,i, :] = (pi[i]) + (phi[i, x[0]])
    
    
    for t in range(1, n):
        for i in range(0,m):
            for j in range(0, k):
                g[t,i,j] = logSumExp(np.concatenate((g[t-1, :,:].ravel(),
                                                     np.asarray(Tmat[:,i]),np.asarray(psi[i,j,:]))))
    return(g)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.double_t, ndim=3] backwardAlgARHMM(int n, int m, int k, np.ndarray[np.double_t, ndim=1] pi, 
                                                       np.ndarray[np.double_t, ndim=2] Tmat, 
                                                       np.ndarray[np.double_t, ndim=3] psi, 
                                                       long[:] x):
    cdef np.ndarray[np.double_t, ndim=3] r
    r = np.zeros((n,m,k))
    cdef int j, l,i 
    for j in range(n-2, -1, -1):
        for l in range(0, m):
            for i in range(0,k):
                r[j, l, i] = logSumExp(np.concatenate((r[j+1,:,:].ravel(), np.asarray(Tmat[l,:]), psi[:, i, :].ravel())))
    
    return(r)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple first_orderARHMM(int n, int m, int k, np.ndarray[np.double_t, ndim=1] pi, 
                             np.ndarray[np.double_t, ndim=2] Tmat, np.ndarray[np.double_t, ndim=2]phi, 
                             long[:] x, double tol):
    #randomly initialize pi, phi and T#
    cdef np.ndarray[np.double_t, ndim=3] vals = np.random.rand(m,k,k)
    cdef np.ndarray[np.double_t, ndim=3] gamma = np.zeros(shape = (n, m,k))
    cdef np.ndarray[np.double_t, ndim=3] g = np.zeros(shape = (n,m,k))
    cdef np.ndarray[np.double_t, ndim=3] h = np.zeros(shape = (n,m,k))
    cdef np.ndarray[np.double_t, ndim=3] psi = np.zeros(shape = (m,k,k))
    cdef int i, t, j, w
    cdef int iterations = 0, convergence = 0, count = 0
    cdef double pOld = 1E10
    cdef double pNew = 0
    cdef double[:] indicies 
    cdef double criteria = 0
    
    psi = np.log(vals/np.sum(vals, axis = 1)[:,None, :])
    
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        #Perform forward and backward algorithms# 
        g = forwardAlgARHMM(n, m, k, pi, Tmat, phi, psi, x)
        h = backwardAlgARHMM(n, m, k, pi, Tmat, psi, x)
        pNew = pForwardARHMM(g, x)
        ##E-Step##
    
        #Calculate gamma and beta#
        for t in range(0, n):
            for i in range(0,m):
                for j in range(0,k):
                    gamma[t,i,j] = g[t,i,j] + h[t,i,j] - pNew
        
        ##M-Step##
    
        #Update psi#
        
        
        for i in range(0,m):
            for q in range(0,k):
                for w in range(0, k):
                    j = 0
                    count = 0
                    for t in range(0,n):
                        if x[t] == w:
                            count = count+1
                    indicies = np.zeros(count)
                    for t in range(0,n):
                        if x[t] == w:
                            indicies[j] = gamma[t,i,q]
                            j = j+1

                    psi[i,w,q] = logSumExp(indicies) - logSumExp(gamma[:,i,q])
        
        criteria = abs(pOld - pNew)
        if criteria < tol or iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
            print(iterations)
    return (iterations, pNew, np.exp(psi))


## NSHMM ##


from scipy.stats import poisson
from collections import Counter

def trans_mat(q, m):
    b = np.zeros((m,m))
    for (x,y), c in Counter(zip(q, q[1:])).items():
        b[x-1,y-1] = c
    return(b) 

from scipy.stats import poisson


def emission_count(q, x, m, k):
    output = np.zeros((m,k))
    for i in range(m):
        for j in range(k):
            output[i,j]=np.where(x[np.where(q == i)[0]] == j)[0].shape[0]
    return(output)
 

import cython
import numpy as np
cimport numpy as np
from scipy.stats import poisson

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.int64_t, ndim=2] MCMC(int n, int m, int k, np.ndarray[np.int64_t, ndim=1] x, double tol, int num_it):
    #Assume uninformative prior
    cdef np.ndarray[np.double_t, ndim=1] alpha = np.ones(m)
    cdef np.ndarray[np.double_t, ndim=1] delta = np.zeros(m)
    cdef np.ndarray[np.double_t, ndim=1] m_mat = np.zeros(m)
    cdef np.ndarray[np.double_t, ndim=2] w = np.zeros((m,m))
    cdef np.ndarray[np.double_t, ndim=2] n_mat = np.zeros((m,m))
    cdef np.ndarray[np.double_t, ndim=2] mstar = np.zeros((m,k))
    cdef np.ndarray[np.double_t, ndim=2] eta = np.ones((m,m))
    cdef np.ndarray[np.double_t, ndim=2] b = np.zeros((m,k))
    cdef np.ndarray[np.double_t, ndim=2] gamma = np.ones((m,k))
    cdef np.ndarray[np.int64_t, ndim=1] q = np.zeros(n, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] output = np.zeros((n,num_it), dtype=np.int64)
    cdef int count = 0, u, v, i, countq = 0, temp = 0
 
    u = 1
    v = 1
    #assume q initially random
    q = np.random.randint(m, size=n)
    while count < tol:
        delta = np.zeros(m)
        delta[q[0]] = 1
        pi = np.random.dirichlet(alpha+delta)
        #print('pi done')
        n_mat = trans_mat(q, m)
        mstar = emission_count(q, x, m, k)
        for i in range(m):
            w[i,:] = np.random.dirichlet(eta[i,:] + n_mat[i,:])
            b[i,:] = np.random.dirichlet(gamma[i,:] + mstar[i,:])
        
        #print('B and T done')
        
        h = backwardAlg(n, m, k, np.log(pi), np.log(w), np.log(b), x)

        q[0] = np.random.choice(range(m), size = 1, p = pi)
        for t in range(1, n):
            prob = np.asarray(h[t,:]) + np.asarray(np.log(b[:,x[t]])) + np.asarray(np.log(w[q[t-1],:]))
            prob = prob/np.sum(prob)
            q[t] = np.random.choice(range(m), size = 1, p = prob)
        

        count+=1
        if (count % 100==0):
            print(count)
            
    for t in range(num_it):
        for i in range(n):
            output[i, t] = np.random.choice(range(k), size = 1, p = b[q[i],:])
    return output
