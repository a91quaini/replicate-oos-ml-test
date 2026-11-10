"""Population checks and mandatory learner preflight; no production outcomes used."""
import argparse
import numpy as np
import design

def scientific():
    for dgp in ('structured','simple'):
        np.testing.assert_allclose(design.moments(dgp,24),design.moments(dgp,48),atol=2e-14)
    cells=design.grid();assert len(cells)==7 and len(design.grid(True))==3
    for c in cells:
        pop=design.population(c['n'],c['delta'],c['dgp']);a,b=pop['a'],pop['b']
        (v1,t1),(v2,t2)=design.moments(c['dgp']);mu=design.MU;tau=design.TAU
        # Integrate the Bernoulli risk polynomial, independently of the root's expression.
        def risk(scale,v,t):
            m1=mu;m2=mu**2+scale**2*v;m3=mu**3+3*mu*scale**2*v+scale**3*t
            return (1+tau)*m1-(1+2*tau)*m2+tau*m3
        np.testing.assert_allclose(risk(a,v1,t1)-risk(b,v2,t2),pop['theta'],atol=2e-14)
        m=design.meta(c)
        for dates in np.array_split(np.arange(c['n']),c['refits']):assert m['origin']+dates[0]-m['gap']-c['T']>=0
        assert c['T']-int(np.ceil(c['T']/4))-m['gap']>20
    m=np.linspace(.03,.97,111);f=m+.005*np.cos(np.arange(len(m)))
    expected=lambda ff:m*design.score(np.ones(len(m)),ff)+(1-m)*design.score(np.zeros(len(m)),ff)
    raw=lambda ff:m*design.loss(np.ones(len(m)),ff)+(1-m)*design.loss(np.zeros(len(m)),ff)
    np.testing.assert_allclose(expected(m),raw(m),atol=1e-14)
    np.testing.assert_allclose((expected(m+1e-5)-expected(m-1e-5))/2e-5,0,atol=1e-8)
    e=f-m
    remainder=(1+2*design.TAU-3*design.TAU*m)*e**2-2*design.TAU*e**3
    np.testing.assert_allclose(expected(f)-expected(m),remainder,atol=1e-14)
    rng=np.random.default_rng(573);z=rng.uniform(-1,1,(120000,5))
    for dgp in ('structured','simple'):
        for j,(v,t) in enumerate(design.moments(dgp)):
            h=design.signal(z,j,dgp);assert abs(h.mean())<6*h.std()/np.sqrt(len(h))
            assert abs((h*h).mean()-v)<6*(h*h).std()/np.sqrt(len(h))
    c=cells[1];a=design.data(c,13);b=design.data(cells[2],13)
    np.testing.assert_array_equal(a['y'],b['y'])
    for j in (0,1):np.testing.assert_array_equal(a['x'][j],b['x'][j])
    print('Scientific checks passed: exact population risks, probability bounds, target preservation, orthogonality, remainder identity, chronology and pairing.')

def torch_checks(device):
    import torch
    from network import optimize,fit
    from experiment import simulate
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    rng=np.random.default_rng(18);x=rng.uniform(-1,1,(2,40,5));y=rng.integers(2,size=(2,40)).astype(float)
    model,d,_=optimize(x,y,[.01,.03],[123,456],2,8,5,device,horizons=[2,5])
    single,_,_=optimize(x[:1],y[:1],[.01],[123],2,8,5,device,horizons=[2])
    with torch.no_grad():
        a=model(torch.as_tensor(x,dtype=torch.float32,device=device)).cpu().numpy()
        b=single(torch.as_tensor(x[:1],dtype=torch.float32,device=device)).cpu().numpy()
        np.testing.assert_allclose(a[0],b[0],atol=4e-6,rtol=4e-6)
    assert (a>.0099).all() and (a<.9901).all()
    assert sum(v.numel() for v in single.parameters())==design.parameters(design.grid()[0])
    assert np.all(d['objective']<=d['initial_objective']+1e-7) and d['best_epoch'][0]<=2
    # Test actual checkpoint selection at 100, as well as the reduced final checkpoint.
    c=dict(design.grid()[0],n=100,T=120,refits=1)
    xx=rng.uniform(-1,1,(2,120,5));yy=rng.integers(2,size=(2,120)).astype(float)
    _,diags=fit(xx,yy,c,[1,2],0,0,device,epochs=101)
    assert all(v['epochs'] in (100,101) and v['selected_kappa']>0 for v in diags)
    result=simulate(c,[1,2],device,epochs=2)
    assert all(len(r['series']['scores']['oracle_raw'])==100 for r in result)
    original=simulate(c,[1],device,epochs=2)[0]
    altered=design.data(c,1);altered['y'][design.meta(c)['origin']:]=1-altered['y'][design.meta(c)['origin']:]
    changed=simulate(c,[1],device,epochs=2,override=[altered])[0]
    assert original['prediction_sha256']==changed['prediction_sha256']
    print('PyTorch checks passed: independent batched fits, variable epoch horizons, validation selection, bounded outputs, full score pipeline and no evaluation leakage.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--torch',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    scientific()
    if a.torch:torch_checks(a.device)
