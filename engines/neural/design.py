"""Five-input nonlinear conditional probabilities, exact polynomial risk calibration."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

SEED=2026111301
BATCH=32
STARTS=2
KAPPAS=(.003,.03,.3,3.)
EPOCHS=600
CHECKPOINTS=(100,200,400,600)
MU=.25
TAU=4.

def grid(pilot=False):
    settings=[(500,4000,0.,'structured'),(1000,4000,0.,'structured'),(1000,8000,0.,'structured')]
    if not pilot:settings.extend((1000,4000,d,'structured') for d in (-.2,-.1,.1,.2))
    return [dict(id=i,n=n,T=T,delta=d,dgp=g,p=5,depth=2,width=8,epochs=EPOCHS,
                 refits=5,schedule='infrequent',tuning='validation',rho=.5)
            for i,(n,T,d,g) in enumerate(settings)]

def components(z,rule):
    x=z[...,0];u=z[...,1];v=z[...,2];s=z[...,3];t=z[...,4]
    interaction=(9*u*u*v*v-1)/8
    if rule==0:return ((3*x*x-1)/2,interaction,.75*(s*s+t*t)-.5)
    return ((5*x**4-1)/4,interaction,.3*(s+t)**2-.2)

def signal(z,rule,dgp='structured'):
    if dgp=='simple':
        return (3*z[...,0]**2-1)/2 if rule==0 else (9*z[...,0]**2*z[...,1]**2-1)/8
    return sum(w*h for w,h in zip((.5,.3,.2),components(z,rule)))

@lru_cache(None)
def moments(dgp='structured',order=32):
    x,w=np.polynomial.legendre.leggauss(order);w=w/2
    q=(3*x*x-1)/2;q4=(5*x**4-1)/4
    h=(9*x[:,None]**2*x[None,:]**2-1)/8
    r=.75*(x[:,None]**2+x[None,:]**2)-.5
    s=.3*(x[:,None]+x[None,:])**2-.2;ww=w[:,None]*w[None,:]
    def m(v,weight):return [float(np.sum(weight*v**k)) for k in (1,2,3)]
    mq,mq4,mh,mr,ms=m(q,w),m(q4,w),m(h,ww),m(r,ww),m(s,ww)
    if dgp=='simple':return (mq[1:],mh[1:])
    result=[]
    for terms in ((mq,mh,mr),(mq4,mh,ms)):
        assert all(abs(t[0])<1e-12 for t in terms)
        # Independent, centered components: cross terms vanish through order 3.
        result.append([sum(a**k*t[k-1] for a,t in zip((.5,.3,.2),terms)) for k in (2,3)])
    return tuple(result)

def signal_bounds(dgp):
    return ((-.3875,1.),(-.2025,1.)) if dgp=='structured' else ((-.5,1.),(-.125,1.))

@lru_cache(None)
def population(n,delta,dgp):
    (v1,t1),(v2,t2)=moments(dgp);a=.24 if dgp=='structured' else .16
    K=1+2*TAU-3*TAU*MU;theta=delta/np.sqrt(n)
    target=K*a*a*v1-TAU*a**3*t1+theta
    b=brentq(lambda b:K*b*b*v2-TAU*b**3*t2-target,1e-8,.95-MU-a,xtol=1e-14)
    bounds=signal_bounds(dgp)
    lower=MU+a*bounds[0][0]+b*bounds[1][0];upper=MU+a+b
    if not .02<=lower<upper<=.95:raise ValueError('Probability support invalid')
    return dict(a=a,b=b,theta=float(theta),lower=lower,upper=upper,signal_sd=[a*np.sqrt(v1),b*np.sqrt(v2)])

def meta(c):
    gap=int(np.ceil(2*np.log(c['n'])/abs(np.log(c['rho']))))
    return dict(gap=gap,origin=8*c['n']+gap,bandwidth=max(gap,int(np.ceil(c['n']**.25))))

def truth(x,c):
    pop=population(c['n'],c['delta'],c['dgp'])
    m=np.stack([MU+pop['a']*signal(x[0],0,c['dgp']),MU+pop['b']*signal(x[1],1,c['dgp'])])
    return m[0]+m[1]-MU,m

def data(c,rep):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,c['n'],rep,1]))
    size=meta(c)['origin']+c['n'];z=np.empty((size,10));z[0]=rng.normal(size=10)
    eps=rng.normal(size=(size-1,10))
    for t in range(1,size):z[t]=c['rho']*z[t-1]+np.sqrt(1-c['rho']**2)*eps[t-1]
    z=2*ndtr(z)-1;x=(z[:,:5],z[:,5:]);p,m=truth(x,c)
    return dict(x=x,y=(rng.uniform(size=size)<p).astype(float),m=m)

def diagnostic(c,rep):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,c['n'],rep,99]))
    z=rng.uniform(-1,1,(4096,10));x=(z[:,:5],z[:,5:]);p,m=truth(x,c)
    return dict(x=x,m=m,probability=p)

def alpha(f):return -2*TAU*f*(1-f)
def loss(y,f):return (1+TAU*y)*(y-f)**2
def score(y,f):return loss(y,f)+alpha(f)*(y-f)
def scores(y,f,m):
    raw=loss(y,f[0])-loss(y,f[1])
    return dict(raw=raw,orthogonal=score(y,f[0])-score(y,f[1]),
                oracle_raw=loss(y,m[0])-loss(y,m[1]),oracle=score(y,m[0])-score(y,m[1]),
                oracle_alpha=raw+alpha(m[0])*(y-f[0])-alpha(m[1])*(y-f[1]))

def parameters(c):
    p,w,d=c['p'],c['width'],c['depth']
    return (p+1)*w+(d-1)*(w*w+w)+w+p+1
