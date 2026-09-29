"""Differential operators and analytic fields for the periodic HJB/HJI solver."""


from __future__ import annotations


import argparse,copy,hashlib,json,math,os,sys,time


from dataclasses import asdict,dataclass,replace


from datetime import datetime,timezone


from pathlib import Path


os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


import numpy as np


import torch


from torch import nn


HERE=Path(__file__).resolve().parent


METHODS=('AD_direct','FD_direct','AD_PI','FD_PI')


@dataclass(frozen=True)
class Config:
    dimension:int=5
    steps:int=10000
    batch_size:int=1024
    width:int=64
    depth:int=3
    policy_interval:int=200
    log_interval:int=500
    learning_rate:float=1e-3
    final_lr_ratio:float=.05
    horizon:float=.5
    viscosity:float=.1
    minimizing_speed:float=.4
    maximizing_speed:float=.3
    h:float=.05
    tau:float=.0025
    monitor_samples:int=4096
    final_samples:int=32768
    evaluation_batch:int=1024
    dtype:str='float64'
    device:str='cuda'


def drift(x):return .15*torch.sin(torch.roll(x,-1,-1))+.05*torch.sin(torch.roll(x,-2,-1)-x)


def diffusion(cfg,like):
    diag=like.new_full((cfg.dimension,),cfg.viscosity);diag[::2]+=.01
    return diag


def hamiltonian(p,gradient,cfg,Q,controls=None):
    projection=gradient@Q
    nonlinear=-cfg.minimizing_speed*projection[:,:2].abs().sum(-1)+cfg.maximizing_speed*projection[:,2:].abs().sum(-1)
    return (drift(p[:,1:])*gradient).sum(-1)+(nonlinear if controls is None else (controls*projection).sum(-1))


def exact_fields(p,cfg,Q):
    t=p[:,0];x=p[:,1:];d=cfg.dimension;s=cfg.horizon-t
    sx,cx=torch.sin(x),torch.cos(x);phase=x.sum(-1);ss,cs=torch.sin(phase),torch.cos(phase)
    g=.3*cx.mean(-1)+.1*cs
    adjacent=torch.roll(sx,-1,-1)+torch.roll(sx,1,-1)
    psi=(sx*torch.roll(sx,-1,-1)).mean(-1)+.4*ss
    gx=-.3*sx/d-.1*ss[:,None];gxx=-.3*cx/d-.1*cs[:,None]
    px=cx*adjacent/d+.4*cs[:,None];pxx=-sx*adjacent/d-.4*ss[:,None]
    rate=2*math.pi/cfg.horizon
    amplitude=s+.05*(1-torch.cos(rate*s));amplitude_s=1+.05*rate*torch.sin(rate*s)
    value=g+amplitude*psi;vt=-amplitude_s*psi
    gradient=gx+amplitude[:,None]*px;second=gxx+amplitude[:,None]*pxx
    cost=-vt-hamiltonian(p,gradient,cfg,Q)-.5*(diffusion(cfg,p)*second).sum(-1)
    return dict(value=value,vt=vt,gradient=gradient,second=second,cost=cost,g=g)


def ad(model,p,training):
    q=p.detach().clone().requires_grad_(True);value=model(q)
    first=torch.autograd.grad(value.sum(),q,create_graph=True)[0]
    second=torch.stack([torch.autograd.grad(first[:,j].sum(),q,retain_graph=True,create_graph=training)[0][:,j] for j in range(1,p.shape[1])],-1)
    return value,first,second


def fd(model,p,cfg):
    n=cfg.dimension;offsets=p.new_zeros((2*n+2,n+1));offsets[1,0]=-cfg.tau
    for j in range(n):offsets[2+2*j,1+j]=cfg.h;offsets[3+2*j,1+j]=-cfg.h
    vals=model((p[None]+offsets[:,None]).reshape(-1,n+1)).reshape(2*n+2,-1)
    first=[(vals[0]-vals[1])/cfg.tau];second=[]
    for j in range(n):
        vp,vm=vals[2+2*j],vals[3+2*j];first.append((vp-vm)/(2*cfg.h));second.append((vp-2*vals[0]+vm)/cfg.h**2)
    return vals[0],torch.stack(first,-1),torch.stack(second,-1)


def policy(model,p,kind,cfg,Q):
    if kind=='AD':
        q=p.detach().clone().requires_grad_(True);gradient=torch.autograd.grad(model(q).sum(),q)[0][:,1:].detach()
    else:
        with torch.no_grad():
            n=cfg.dimension;offsets=p.new_zeros((2*n,n+1))
            for j in range(n):offsets[2*j,1+j]=cfg.h;offsets[2*j+1,1+j]=-cfg.h
            vals=model((p[None]+offsets[:,None]).reshape(-1,n+1)).reshape(2*n,-1)
            gradient=torch.stack([(vals[2*j]-vals[2*j+1])/(2*cfg.h) for j in range(n)],-1)
    projection=gradient@Q
    return torch.cat([-cfg.minimizing_speed*torch.sign(projection[:,:2]),cfg.maximizing_speed*torch.sign(projection[:,2:])],-1)


def sample(n,g,cfg):
    p=torch.rand((n,cfg.dimension+1),generator=g,device=cfg.device,dtype=getattr(torch,cfg.dtype))
    p[:,0]*=cfg.horizon;p[:,1:]*=2*math.pi
    return p


def state_hash(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def sync(cfg):
    if cfg.device=='cuda':torch.cuda.synchronize()
