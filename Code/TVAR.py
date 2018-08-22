import scipy
import numpy as np

# Fit TVAR(p) model to univariate series x
# Discount factors del(1) for state and del(2) for obs var
# Inputs:
#   x --  series of length T
#   p --  model order
#  m0 --  px1 vector prior mean for state
#  C0 --  pxp prior var matrix
#  n0 --  prior df
#  s0 --  prior estimate of obs var
# Outputs: 
#  forward filtered: 
#  mf  --  pxT   post mean vectors
#  Cf  --  pxpxT post var matrices
#  nf  --  T post dfs
#  sf  --  T post obs var estimates
#  ef  --  1-step forecast errors (zero to t=p) 
#  qf  --  1-step forecast variance factors
#  retrospectively updated/smoothed: 
#  m   --  pxT   post mean vectors
#  C   --  pxpxT post var matrices
#  n   --  T post dfs
#  s   --  T post obs var estimates
#  e   --  estimated innovations (zero to t=p) 
# 

def tvar(x,p,delta,m0,C0,s0,n0):
    d=delta[0] 
    b = delta[1]
    arx=x-np.mean(x)
    T=len(arx)
    arx = np.transpose(arx)
    #arx=reshape(arx,T,1);
    m = np.repeat(m0.reshape(p,1), T, axis=1)
    C = np.repeat(C0[:, :, np.newaxis], T, axis=2)
    s = np.repeat(s0,T)
    n = np.repeat(n0,T)
    e = np.zeros(shape = (T,1))
    q = np.zeros(shape = (T,1))
    mt=m0; Ct=C0; st=s0; nt=n0;

    ## forward filtering ##
    for t in range(p,T):
        if t == p:
            F = arx[t-1::-1].reshape(p,1)
        else:
            F = arx[t-1:t-p-1:-1].reshape(p,1)
        A = np.dot(Ct,F)/d
        qt = np.dot(F.transpose(),A) + st
        qt = qt.flatten()
        A = A/qt
        et = arx[t] - np.dot(F.transpose(),mt)
        et = et.flatten()
        e[t] = et
        q[t] = qt
        mt = mt+et*A
        m[:,t] = mt.flatten()
        r=b*nt+et*et/qt 
        nt=b*nt+1
        r=r/nt
        st=st*r
        n[t]=nt 
        s[t]=st
        Ct=r*(Ct/d-np.dot(A,A.transpose())*qt) 
        Ct=(Ct+Ct.transpose())/2;
        C[:,:,t]=Ct
     
    ## save filtered values ...
    mf=m; Cf=C; sf=s; nf=n; ef=e; qf=q; 

    ## backward smoothing
    for t in range((T-2),-1,-1):
        m[:,t] = (1-d)*m[:,t] + d*m[:, t+1]
        if t>p:
            e[t]=arx[t]-np.dot(m[:,t].transpose(),arx[t-1:t-p-1:-1])
        n[t] = (1-b)*n[t]+b*n[t+1]  
        st=s[t] 
        s[t]=1/((1-b)/st+b/s[t+1]) 
        C[:,:,t]=s[t]*((1-d)*C[:,:,t]/st + d*d*C[:,:,t+1]/s[t+1]); 
     
    return (m,C,n,s,e,mf,Cf,sf,nf,ef,qf)



# Compute lik fn for TVAR model orders 1,...,p 
# Inputs:
#   x --  series of length T
# pvals -- [pmin,pmax] -- range for model order
# ndel -- number of discount factors to consider in range 0.95-1
#  m0 --  px1 vector prior mean for state
#  C0 --  pxp prior var matrix
#  n0 --  prior df
#  s0 --  prior estimate of obs var
# Outputs: 
# likp  --  px ndel x ndel array with log-lik fn
# popt   -- MLE of model order
# delopt -- MLEs of discounts 

def tvar_lik(x,pvals,dn,bn,m0,C0,s0,n0):
    ndel = np.array([len(dn), len(bn)]) # AR and v discounts 
    arx=x-np.mean(x)
    T=len(arx)
    arx = np.transpose(arx)
    pmax=pvals[1] 
    pmin=pvals[0]
    likp=np.zeros(shape = (pmax-pmin+1,ndel[0],ndel[1])) 
    popt=0; dopt=1; bopt=1; maxlik=-1e300;
    for p in range(pmin, pmax+1,1):
        for i in range(ndel[0]):
            d=dn[i];
            for j in range(ndel[1]):
                b=bn[j]
                mt=m0[0:p] 
                Ct=C0[0:p,0:p] 
                st=s0 
                nt=n0
                llik=0;
                for t in range(p,T):
                    if t == p:
                        F = arx[t-1::-1].reshape(p,1)
                    else:
                        F = arx[t-1:t-p-1:-1].reshape(p,1)
                    A=np.dot(Ct,F)/d 
                    q=np.dot(F.transpose(), A)+st 
                    A=A/q 
                    f=np.dot(F.transpose(),mt) 
                    e=arx[t]-f 
                    nt=b*nt;
                    if t>2*pmax:   # ignore first 2*pmax observations for comparison
                        llik=llik+scipy.special.gammaln((nt+1)/2)-scipy.special.gamma(nt/2)-np.log(nt*q)/2- \
                        (nt+1)*np.log(1+e*e/(q*nt))/2 
                     
                    mt=mt+A*e; r=nt+e*e/q; nt=nt+1; r=r/nt; st=st*r; 
                    Ct=r*(Ct/d-np.dot(A,A.transpose())*q); 
             
                likp[p-pmin,i,j]=llik
                if (llik>maxlik):
                    popt=p; dopt=d; bopt=b; maxlik=llik;

    delopt=[dopt,bopt];
    return (popt,delopt,likp)


# Sample posteriors in TVAR model
# 
# Inputs:
#  m  --  pxT array of post means for states
#  C  --  pxpxT post var matrices
#  n  --  T post dfs
#  s  --  T post obs var estimate
#  times  --  vector of times to sample, length nt 
#  N  --  MC sample size
#
# Output: 
#   phis   -- p x nt x N array of post sampled \phi_t
#   freqs  -- k x nt x N array of post sampled TVAR comp frequencies
#   
 
def tvar_sim(m,C,n,s,times,N):
    nt=len(times) 
    (p,T)=m.shape;
    phis=np.zeros(shape = (p,nt,N))

    # sample frequencies ...
    for it in range(nt):
        t=it
        x = np.dot(np.linalg.cholesky(C[:,:,t]).transpose(), 
                                       np.random.normal(0,1,(p,N)))
        y = np.tile(np.sqrt(np.random.gamma(n[t]/2,2/n[t],N)), (p,1))
        theta=np.array([m[:,t],]*N).transpose()+x/y
        phis[:,it,:]=theta; 
    return phis
