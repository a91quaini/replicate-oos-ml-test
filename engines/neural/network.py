"""Independent networks batched on a GPU; no pooling across Monte Carlo draws."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from design import SEED, STARTS, KAPPAS, CHECKPOINTS

class Ensemble(nn.Module):
    def __init__(self,p,depth,width,seeds,device='cpu'):
        super().__init__();self.p=p
        rngs=[np.random.default_rng(s) for s in seeds]
        def parameter(shape,scale,zero=False):
            a=np.stack([np.zeros(shape) if zero else r.normal(0,scale,shape) for r in rngs])
            return nn.Parameter(torch.tensor(a,dtype=torch.float32,device=device))
        self.weights=nn.ParameterList();self.biases=nn.ParameterList()
        for i in range(depth):
            fan=p if i==0 else width
            self.weights.append(parameter((fan,width),np.sqrt(2/fan)))
            self.biases.append(parameter((1,width),0,True))
        self.head=parameter((width,1),.05/np.sqrt(width))
        self.skip=parameter((p+1,1),0,True)
        with torch.no_grad():
            for v in self.parameters():v.clamp_(-3,3)
    def forward(self,x,diagnostics=False):
        z=x;dead=[]
        for w,b in zip(self.weights,self.biases):
            z=torch.relu(torch.bmm(z,w)+b)
            if diagnostics:dead.append((z.amax(dim=1)==0).float().mean(dim=1))
        linear=torch.bmm(z,self.head)+torch.bmm(x,self.skip[:,:-1])+self.skip[:,-1:]
        y=(.5+.49*torch.tanh(linear)).squeeze(-1)
        return (y,torch.stack(dead,dim=1)) if diagnostics else y
    def penalty(self):
        ans=self.head.square().sum((1,2))+self.skip[:,:-1].square().sum((1,2))
        for w,b in zip(self.weights,self.biases):ans=ans+w.square().sum((1,2))+b.square().sum((1,2))
        return ans/2


def optimize(x,y,lambdas,seeds,depth,width,epochs,device,horizons=None,validation=None):
    model=Ensemble(x.shape[-1],depth,width,seeds,device)
    xx=torch.as_tensor(x,dtype=torch.float32,device=device);yy=torch.as_tensor(y,dtype=torch.float32,device=device)
    lam=torch.as_tensor(lambdas,dtype=torch.float32,device=device)
    horizon_array=np.full(len(x),epochs) if horizons is None else np.asarray(horizons)
    horizon_steps=set(int(v) for v in horizon_array)
    horizon=torch.as_tensor(horizon_array,device=device)
    if bool(((horizon<1)|(horizon>epochs)).any()):raise ValueError('Invalid epoch horizons')
    with torch.no_grad():
        model.skip[:,-1,0]=torch.atanh(((yy.mean(1)-.5)/.49).clamp(-.98,.98))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
    best=torch.full((len(x),),float('inf'),device=device)
    snapshot=[v.detach().clone() for v in model.parameters()];best_step=torch.zeros(len(x),device=device)
    initial=None;trace=[]
    vx,vy=(None,None) if validation is None else [torch.as_tensor(v,dtype=torch.float32,device=device) for v in validation]
    checks=set([e for e in CHECKPOINTS if e<=epochs]+[epochs])
    def capture(step):
        nonlocal initial,best
        with torch.no_grad():
            loss=.5*((model(xx)-yy)**2).mean(1)+lam*model.penalty()
            if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite objective; draw retained as failure')
            if initial is None:initial=loss.clone()
            mask=(loss<best)&(step<=horizon)
            best=torch.where(mask,loss,best);best_step[mask]=step
            for dst,src in zip(snapshot,model.parameters()):dst.copy_(torch.where(mask[:,None,None],src,dst))
            if validation is not None and step in checks:
                current=[v.detach().clone() for v in model.parameters()]
                for v,b in zip(model.parameters(),snapshot):v.copy_(b)
                val=((model(vx)-vy)**2).mean(1).cpu().numpy()
                trace.append(dict(epoch=step,objective=best.cpu().numpy().copy(),validation_mse=val))
                for v,b in zip(model.parameters(),current):v.copy_(b)
    capture(0)
    for step in range(int(horizon.max().item())):
        optimizer.param_groups[0]['lr']=float(.0003+.0027*.5*(1+np.cos(np.pi*step/max(epochs-1,1))))
        optimizer.zero_grad(set_to_none=True)
        loss=.5*((model(xx)-yy)**2).mean(1)+lam*model.penalty()
        loss.sum().backward();optimizer.step()
        with torch.no_grad():
            for v in model.parameters():v.clamp_(-3,3)
        if (step+1)%10==0 or step+1 in checks or step+1 in horizon_steps:capture(step+1)
    with torch.no_grad():
        for v,b in zip(model.parameters(),snapshot):v.copy_(b)
    optimizer.zero_grad(set_to_none=True)
    prediction=model(xx);objective=.5*((prediction-yy)**2).mean(1)+lam*model.penalty()
    objective.sum().backward()
    mapping=sum(((v-(v-v.grad).clamp(-3,3))**2).sum((1,2)) for v in model.parameters()).sqrt()
    with torch.no_grad():
        pred,dead=model(xx,True)
        diag=dict(objective=objective.cpu().numpy(),initial_objective=initial.cpu().numpy(),
                  training_mse=((pred-yy)**2).mean(1).cpu().numpy(),
                  projected_gradient=mapping.cpu().numpy(),best_epoch=best_step.cpu().numpy(),
                  dead_units=dead.cpu().numpy(),saturation=((pred<.011)|(pred>.989)).float().mean(1).cpu().numpy())
    return model,diag,trace


def fit(x,y,c,reps,rule,date,device,epochs=None):
    from design import meta
    B,T,p=x.shape;steps=epochs or c['epochs'];ks=list(reversed(KAPPAS));C=len(ks)
    split=T-int(np.ceil(T/4));prefix=split-meta(c)['gap']
    if prefix<20:raise ValueError('Insufficient fitting prefix')
    def seeds(phase,count):
        return [np.random.SeedSequence([SEED,c['n'],r,rule,date,phase,s]) for r in reps for _ in range(count) for s in range(STARTS)]
    repeat=C*STARTS
    candidate,_,trace=optimize(np.repeat(x[:,:prefix],repeat,0),np.repeat(y[:,:prefix],repeat,0),
        np.tile(np.repeat(np.array(ks)/np.sqrt(T),STARTS),B),seeds(0,C),c['depth'],c['width'],steps,device,
        validation=(np.repeat(x[:,split:],repeat,0),np.repeat(y[:,split:],repeat,0)))
    # Candidate order: stronger penalty first, then earlier epoch; starts chosen by training objective.
    validation=np.empty((B,C,len(trace)))
    for e,t in enumerate(trace):
        starts=t['objective'].reshape(B,C,STARTS).argmin(2)
        val=t['validation_mse'].reshape(B,C,STARTS)
        validation[:,:,e]=np.take_along_axis(val,starts[:,:,None],axis=2).squeeze(2)
    flat=validation.reshape(B,-1);chosen=np.argmax(flat<=flat.min(1,keepdims=True)+1e-12,axis=1)
    ki,ei=np.unravel_index(chosen,(C,len(trace)))
    kappas=np.array(ks)[ki];selected_epochs=np.array([trace[e]['epoch'] for e in ei]);del candidate
    model,diag,_=optimize(np.repeat(x,STARTS,0),np.repeat(y,STARTS,0),np.repeat(kappas/np.sqrt(T),STARTS),
        seeds(1,1),c['depth'],c['width'],steps,device,horizons=np.repeat(selected_epochs,STARTS))
    best_start=diag['objective'].reshape(B,STARTS).argmin(1)+np.arange(B)*STARTS
    result=[]
    for i,j in enumerate(best_start):
        d={k:np.asarray(v[j]).tolist() for k,v in diag.items()}
        d.update(selected_kappa=float(kappas[i]),penalty=float(kappas[i]/np.sqrt(T)),epochs=int(selected_epochs[i]),
                 validation_mse=float(flat[i,chosen[i]]),candidate_validation_mse=validation[i].tolist(),
                 candidate_kappas=ks,candidate_epochs=[t['epoch'] for t in trace],prefix_length=prefix,
                 validation_start=split,validation_constant_mse=float(np.mean((y[i,split:]-y[i,:prefix].mean())**2)),
                 all_start_objectives=diag['objective'][i*STARTS:(i+1)*STARTS].tolist())
        result.append(d)
    def predict(z):
        with torch.no_grad():
            zz=torch.as_tensor(np.repeat(z,STARTS,0),dtype=torch.float32,device=device)
            return model(zz).cpu().numpy()[best_start].astype(float)
    return predict,result

