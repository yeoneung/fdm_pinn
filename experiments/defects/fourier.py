"""Controlled inexact policy iteration on a periodic manufactured HJI problem."""


from __future__ import annotations


import argparse


import csv


from dataclasses import asdict, dataclass


from datetime import datetime, timezone


import hashlib


import json


from pathlib import Path


import platform


import sys


import time


import matplotlib


matplotlib.use("Agg")


import matplotlib.pyplot as plt


import numpy as np


import scipy


from scipy.signal import resample


@dataclass(frozen=True)
class Config:
    grid: int = 33
    steps: int = 128
    iterations: int = 12
    horizon: float = 0.5
    viscosity: float = 0.25
    physical_sigma_x: float = 0.1
    control_speed: float = 0.12
    disturbance_speed: float = 0.08
    amplitude: float = 0.08
    decay: float = 0.55
    rho: float = 0.5
    validation_factor: int = 2
    temporal_quadrature: int = 4

    @property
    def ellipticity(self):
        return self.viscosity

    @property
    def f_squared(self):
        return self.control_speed**2 + self.disturbance_speed**2

    @property
    def kappa(self):
        return 3 + 4*self.f_squared/self.ellipticity + 16*self.f_squared/(3*self.rho*self.ellipticity)


class Spatial:
    def __init__(self, size: int, config: Config):
        self.size = size
        self.config = config
        z = np.arange(size) * (2*np.pi/size)
        self.x, self.y = np.meshgrid(z, z, indexing="ij")
        k = np.fft.fftfreq(size, d=1/size)
        self.kx, self.ky = k[:, None], k[None, :]
        self.ax = (config.physical_sigma_x**2 + config.viscosity)/2
        self.ay = config.viscosity/2
        self.diff_multiplier = -self.ax*self.kx**2 - self.ay*self.ky**2
        x, y = self.x, self.y
        self.g = np.cos(x) + 0.5*np.sin(y) + 0.25*np.cos(x+y)
        self.psi = 0.4*np.sin(x)*np.sin(y) + 0.2*np.cos(2*x)
        self.gx = -np.sin(x) - 0.25*np.sin(x+y)
        self.gy = 0.5*np.cos(y) - 0.25*np.sin(x+y)
        self.px = 0.4*np.cos(x)*np.sin(y) - 0.4*np.sin(2*x)
        self.py = 0.4*np.sin(x)*np.cos(y)
        self.dg = self.ax*(-np.cos(x)-0.25*np.cos(x+y)) + self.ay*(-0.5*np.sin(y)-0.25*np.cos(x+y))
        self.dp = self.ax*(-0.4*np.sin(x)*np.sin(y)-0.8*np.cos(2*x)) + self.ay*(-0.4*np.sin(x)*np.sin(y))
        self.forcing = np.sin(2*x+y)
        self.initial_mode = np.sin(x-y)
        self.initial_dx = np.cos(x-y)
        self.initial_dy = -np.cos(x-y)

    def differential(self, values):
        coeff = np.fft.fft2(values, axes=(-2, -1))
        inv = lambda c: np.fft.ifft2(c, axes=(-2, -1)).real
        return inv(1j*self.kx*coeff), inv(1j*self.ky*coeff), inv(self.diff_multiplier*coeff)

    def exact(self, s):
        return self.g + np.asarray(s)[..., None, None]*self.psi

    def cost(self, s):
        s = np.asarray(s)[..., None, None]
        px, py = self.gx+s*self.px, self.gy+s*self.py
        h = -self.config.control_speed*np.abs(px) + self.config.disturbance_speed*np.abs(py)
        return self.psi - self.dg - s*self.dp - h


def interpolate_space(values, size):
    if values.shape[-1] == size:
        return values
    return resample(resample(values, size, axis=-2), size, axis=-1)


def initial_values(spatial, times):
    return spatial.exact(times) + times[:, None, None]*spatial.initial_mode


def evaluate_policy(previous, amplitude, times, spatial):
    """RK4 in reversed time s=T-t, Fourier collocation in periodic space."""
    cfg = spatial.config
    old_x, old_y, _ = spatial.differential(previous)
    values = np.empty_like(previous)
    values[0] = spatial.g
    dt = times[1]-times[0]

    def rhs(value, px, py, cost):
        vx, vy, diffusion = spatial.differential(value)
        alpha = -cfg.control_speed*np.sign(px)
        beta = cfg.disturbance_speed*np.sign(py)
        # Original-time residual r=amplitude*forcing gives -r after time reversal.
        return diffusion + alpha*vx + beta*vy + cost - amplitude*spatial.forcing

    for j in range(cfg.steps):
        sx, sy = old_x[j], old_y[j]
        ex, ey = old_x[j+1], old_y[j+1]
        mx, my = (sx+ex)/2, (sy+ey)/2
        c0 = spatial.cost(times[j])
        cm = spatial.cost(times[j]+dt/2)
        c1 = spatial.cost(times[j+1])
        v = values[j]
        k1 = rhs(v, sx, sy, c0)
        k2 = rhs(v+dt*k1/2, mx, my, cm)
        k3 = rhs(v+dt*k2/2, mx, my, cm)
        k4 = rhs(v+dt*k3, ex, ey, c1)
        values[j+1] = v + dt*(k1+2*k2+2*k3+k4)/6
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite policy evaluation")
    return values


def initial_energy(cfg):
    # Spatial mean |grad(s*sin(x-y))|^2=s^2; integrate the weighted polynomial.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    s = (nodes+1)*cfg.horizon/2
    return float(cfg.ellipticity*cfg.horizon/2*np.sum(weights*np.exp(-cfg.kappa*s)*s*s))


def self_checks(cfg):
    sp = Spatial(cfg.grid,cfg)
    checks = {}
    s = 0.173
    v = sp.exact(s)
    vx,vy,dv = sp.differential(v)
    checks["analytic_gradient_max_error"] = float(max(np.max(np.abs(vx-sp.gx-s*sp.px)),np.max(np.abs(vy-sp.gy-s*sp.py))))
    residual = -sp.psi-cfg.control_speed*np.abs(vx)+cfg.disturbance_speed*np.abs(vy)+dv+sp.cost(s)
    checks["manufactured_pde_max_residual"] = float(np.max(np.abs(residual)))
    alpha,beta = -cfg.control_speed*np.sign(vx),cfg.disturbance_speed*np.sign(vy)
    attained = alpha*vx+beta*vy
    nested = np.maximum(np.minimum(-cfg.control_speed*vx-cfg.disturbance_speed*vy,cfg.control_speed*vx-cfg.disturbance_speed*vy),
                        np.minimum(-cfg.control_speed*vx+cfg.disturbance_speed*vy,cfg.control_speed*vx+cfg.disturbance_speed*vy))
    checks["nested_control_max_error"] = float(np.max(np.abs(attained-nested)))
    # Independent linear heat equation check tests reverse-time sign and RK stages.
    from dataclasses import replace
    heat_cfg = replace(cfg,control_speed=0.0,disturbance_speed=0.0)
    heat_sp = Spatial(cfg.grid,heat_cfg)
    heat_sp.g = np.cos(heat_sp.x)
    heat_sp.cost = lambda t: np.zeros_like(heat_sp.g)
    times = np.linspace(0,cfg.horizon,cfg.steps+1)
    heat = evaluate_policy(np.zeros((cfg.steps+1,cfg.grid,cfg.grid)),0,times,heat_sp)
    reference = np.exp(-heat_sp.ax*cfg.horizon)*heat_sp.g
    checks["heat_equation_final_max_error"] = float(np.max(np.abs(heat[-1]-reference)))
    checks["passed"] = all(x < 1e-9 for x in checks.values())
    if not checks["passed"]: raise AssertionError(checks)
    return checks
