#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp
import scipy.linalg # For eigh
import scipy.interpolate # !JR ADDED for CubicSpline
from scipy.special import spherical_jn
import matplotlib.pyplot as plt

import time
import concurrent.futures # For k-loop parallelization
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================
# Global Parameters
# ======================================================
pi = np.pi
class StringParams:
    def __init__(self, mu=1.0e-6, alpha=1.9, v=0.65, xi=0.13, L=0.99):
        self.mu=mu; self.alpha=alpha; self.v=v; self.xi=xi; self.L=L
SPRa = StringParams(mu=2.0e-7, alpha=1.9, v=0.65, xi=0.13, L=0.99)

# k_min_calc = 10**-3; k_max_calc = 0.6; nk_calc = 100 # k range
# tau_min_calc = 10**-2; tau_max_calc = 14400.0; ntau_calc = 64 # tau range

# THE RANGE FOR K AND TAU THAT ARE CALCULATED. MAINLY ADJUST THESE
k_min_calc = 1e-8; k_max_calc = 100; nk_calc = 500 # k range
ktau_min_calc = 1e-4; ktau_max_calc = 1e3; nktau_calc = 256 # tau range

# UETC calculation parameters (leave unchanged)
xmin = 0.15; xmax = 20.0; xapr = 2.5; etcmin = 0.001
min_terms = 10; scale_terms = 15.0; MAX_N_TERMS = 75
scaling_option = 2;
uetc_feedback = 0

# Number of eigenmodes
nmodes = nktau_calc
# Weighting of eigenvalue decomposition (gamma in the paper, eq. 32)
weighting = 0.25; # This is 'γ'

# ========================================================================
# Helper Functions AND Integral Functions (I1-I6) 
# ========================================================================

def spher_bessel(n, x):
    x = np.asarray(x); result = np.zeros_like(x, dtype=float); mask_nz = x != 0; mask_z = ~mask_nz
    if n == -1:
        if np.any(mask_nz): result[mask_nz] = np.cos(x[mask_nz]) / x[mask_nz]
        if np.any(mask_z): result[mask_z] = 0.0 # Should be 1/x -> undefined at 0, or limit to 0.
    elif n == -2:
        if np.any(mask_nz): xn = x[mask_nz]; result[mask_nz] = (-np.sin(xn)/xn - np.cos(xn)) / xn
        if np.any(mask_z): result[mask_z] = -1.0/3.0
    elif n >= 0:
        if np.any(mask_nz): result[mask_nz] = sp.spherical_jn(n, x[mask_nz])
        if np.any(mask_z): result[mask_z] = 1.0 if n == 0 else 0.0
    else: result = np.zeros_like(x, dtype=float)
    return np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

def sine_integral(x): si, _ = sp.sici(x); return si

def factorial(n):
    try: return sp.gamma(n + 1.0)
    except ValueError: return np.inf
    

def I1_int(x, rho, n_terms):
    val = 0.0; rho_safe = max(rho, 1e-12); x2_safe = max(x**2, 1e-12)
    base = -x2_safe / (2.0 * rho_safe)
    for i in range(1, int(n_terms) + 1):
        fact_i = factorial(i);
        if fact_i == 0 or fact_i == np.inf or fact_i > 1e300: continue
        try:
             if base == 0: power_term = 0.0 if i > 0 else 1.0
             elif i * abs(np.log(abs(base) if base != 0 else 1)) < 700: power_term = base**i
             else: power_term = np.sign(base**i) * np.inf if base != 0 else 0.0
        except (OverflowError, ValueError, FloatingPointError): power_term = np.sign(base) * np.inf
        if not np.isfinite(power_term): continue
        term_val = (1.0 / fact_i * (rho_safe / (2.0 * i - 1.0)) * power_term * spher_bessel(i - 1, rho_safe))
        if not np.isfinite(term_val): continue
        new_val = val + term_val
        if not np.isfinite(new_val): break
        val = new_val
    return np.nan_to_num(val)


def I4_int(x, rho, n_terms):
    rho_safe = max(rho, 1e-12);
    if rho_safe == 0 : return 0.0
    x2_safe = max(x**2, 1e-12)
    try: val = np.cos(x) / rho_safe**2
    except (ZeroDivisionError, FloatingPointError): val = np.inf * np.cos(x)
    if not np.isfinite(val): val = 0.0
    base = -x2_safe / (2.0 * rho_safe)
    for i in range(1, int(n_terms) + 1):
        fact_i = factorial(i);
        if fact_i == 0 or fact_i == np.inf or fact_i > 1e300: continue
        try:
            if base == 0: power_term = 0.0 if i > 0 else 1.0
            elif i * abs(np.log(abs(base) if base != 0 else 1)) < 700: power_term = base**i
            else: power_term = np.sign(base**i) * np.inf if base != 0 else 0.0
        except (OverflowError, ValueError, FloatingPointError): power_term = np.sign(base) * np.inf
        if not np.isfinite(power_term): continue
        term_val = - (1.0 / fact_i * (1.0 / (2.0 * i - 1.0)) * power_term * spher_bessel(i - 2, rho_safe))
        if not np.isfinite(term_val): continue
        new_val = val + term_val
        if not np.isfinite(new_val): break
        val = new_val
    return np.nan_to_num(val)


def I1_int_a(x, rho):
    if rho == 0: return pi * x / 2.0
    j0_rho = sp.jv(0, rho); return (pi * x / 2.0) * j0_rho
def I4_int_a(x, rho):
    if rho == 0: return pi*x/4.0
    rho_safe = max(rho, 1e-12); j1_rho = sp.jv(1, rho_safe)
    return (pi * x * j1_rho) / (2.0 * rho_safe)
def I2_int(x, rho):
    px = rho**2 + x**2;
    if px < 1e-12: return 1.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); return np.divide(srpx, rpx, out=np.ones_like(srpx), where=rpx!=0)
def I3_int(x, rho):
    px = rho**2 + x**2;
    if px < 1e-12: return -1/3.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); crpx = np.cos(rpx)
    term1_factor = (1.0 - 3.0 * x**2 / px); term2_factor = (1.0 - (1.0 + x**2) / px + 3.0 * x**2 / px**2)
    term1 = np.divide(crpx * term1_factor, px, out=np.zeros_like(px), where=px!=0)
    term2 = np.divide(srpx * term2_factor, rpx, out=np.zeros_like(rpx), where=rpx!=0)
    val = term1 + term2; return np.nan_to_num(val, nan=-1/3.0, posinf=0.0, neginf=0.0)
def I5_int(x, rho):
    rho_safe = max(rho, 1e-12); px = rho_safe**2 + x**2
    if rho_safe**2 < 1e-15 * x**2 and abs(x) > 1e-6: return np.divide(np.sin(x), 2.0*x, out=np.full_like(x, -0.5), where=x!=0)
    elif px < 1e-12: return -0.5
    rpx = np.sqrt(px); crpx = np.cos(rpx); val = (np.cos(x) - crpx) / rho_safe**2
    return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
def I6_int(x, rho):
    px = rho**2 + x**2;
    if px < 1e-12: return 1/3.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); crpx = np.cos(rpx)
    term1 = np.divide(srpx, rpx, out=np.ones_like(srpx), where=rpx!=0)
    val = np.divide(term1 - crpx, px, out=np.full_like(px, 1/3.0), where=px!=0)
    return np.nan_to_num(val, nan=1/3.0, posinf=0.0, neginf=0.0)


def scaling_factor(tau1, tau2, xi1, xi2, L):
    if scaling_option == 1: return 1.0
    elif scaling_option == 2: denom = max(xi1 * tau1, xi2 * tau2, 1e-30); return 1.0 / (denom**3)
    else: return 1.0

def get_uetc(tau1, tau2, k, SPR_base):
    uetc_val = np.zeros(5); SPR1 = SPRa; SPR2 = SPRa
    xi1=SPR1.xi; xi2=SPR2.xi; alpha1=SPR1.alpha; alpha2=SPR2.alpha; mu1=SPR1.mu; mu2=SPR2.mu
    v1=SPR1.v; v2=SPR2.v; L_decay = SPR1.L
    
    if k<=0 or tau1<=0 or tau2<=0 or xi1<=0 or xi2<=0: return uetc_val
    v1=min(v1, 0.9999999); v2=min(v2, 0.9999999)
    x1=k*tau1*xi1; x2=k*tau2*xi2; xp=(x1+x2)/2.0; xm=(x1-x2)/2.0
    rho=k*abs(v1*tau1-v2*tau2); rho_safe=max(rho, 1e-12)
    norm_denom_sq = (1.0 - v1**2)*(1.0 - v2**2)
    if norm_denom_sq <= 1e-16: return uetc_val
    norm_denom = np.sqrt(norm_denom_sq)
    sf = scaling_factor(tau1, tau2, xi1, xi2, L_decay)
    # Note: Common factor does contain mu1*mu2 and sf
    common_factor_base = sf / (k**2*norm_denom) if k!=0 and norm_denom>0 else 0.0
    if not np.isfinite(common_factor_base): return uetc_val
    common_factor = mu1*mu2*common_factor_base
    if not np.isfinite(common_factor) or common_factor == 0: return uetc_val
    

    # --- Regime 1: Small x---
    if x1 <= xmin and x2 <= xmin:
        if uetc_feedback > 1: print('Small x')
        if alpha1==0 or alpha2==0: return uetc_val
        term00=-(alpha1*alpha2*mu1*mu2*(-6.0 + rho**2)*x1*x2)/(6.0*k**2*norm_denom)
        term_s_num=(rho**2*(-10.0+(10.0-11.0*alpha2**2)*v2**2+v1**2*(10.0-11.0*alpha1**2+(-10.0+11.0*alpha1**2+11.0*(1.0-2.0*alpha1**2)*alpha2**2)*v2**2))+42.0*(2.0+(-2.0+alpha2**2)*v2**2+v1**2*(-2.0+alpha1**2+(2.0-alpha1**2+(-1.0+2.0*alpha1**2)*alpha2**2)*v2**2)))*x1*x2
        termS=(mu1*mu2*term_s_num)/(420.0*alpha1*alpha2*k**2*norm_denom) if alpha1*alpha2 != 0 else 0.0
        term_v_num=(8.0+4.0*(-2.0+alpha1**2)*v1**2+(-8.0-4.0*(-2.0+alpha1**2)*v1**2+alpha2**2*(4.0+(-4.0+7.0*alpha1**2)*v1**2))*v2**2)*x1*x2
        termV=(mu1*mu2*2.3*term_v_num)/(256.0*alpha1*alpha2*k**2*norm_denom) if alpha1*alpha2 != 0 else 0.0
        term_t_num=-(mu1*mu2*(-28.0+6.0*rho**2+28.0*v1**2-14.0*alpha1**2*v1**2-6.0*rho**2*v1**2+alpha1**2*rho**2*v1**2+28.0*v2**2-14.0*alpha2**2*v2**2-6.0*rho**2*v2**2+alpha2**2*rho**2*v2**2-28.0*v1**2*v2**2+14.0*alpha1**2*v1**2*v2**2+14.0*alpha2**2*v1**2*v2**2-28.0*alpha1**2*alpha2**2*v1**2*v2**2+6.0*rho**2*v1**2*v2**2-alpha1**2*rho**2*v1**2*v2**2-alpha2**2*rho**2*v1**2*v2**2+2.0*alpha1**2*alpha2**2*rho**2*v1**2*v2**2)*x1*x2)
        termT=term_t_num/(420.0*alpha1*alpha2*k**2*norm_denom) if alpha1*alpha2 != 0 else 0.0
        term_cross_num=-(mu1*mu2*rho**2*(alpha1**2*(1.0+(-1.0+2.0*alpha2**2)*v2**2)+alpha2**2*(1.0+(-1.0+2.0*alpha1**2)*v1**2))*x1*x2)
        term00S=term_cross_num/(60.0*alpha1*alpha2*k**2*norm_denom) if alpha1*alpha2 != 0 else 0.0
        uetc_val=np.array([term00, termS, termV, termT, term00S])*sf
        return np.nan_to_num(uetc_val)

    # --- Regime 2: ETC ---
    if abs(x1 - x2) <= etcmin:
        if uetc_feedback > 1: print('ETC')
        x = xp; alpha=(alpha1+alpha2)/2.0; v=(v1+v2)/2.0; mu=(mu1+mu2)/2.0
        v = min(v, 0.9999999);
        if alpha==0: return np.zeros(5)
        norm_denom_etc_sq = (1.0 - v**2);
        if norm_denom_etc_sq <= 1e-16 : return np.zeros(5)
        norm_denom_etc = np.sqrt(norm_denom_etc_sq)
        
        mu = (mu1 + mu2) / 2.0; tau=(tau1+tau2)/2; xi=(xi1+xi2)/2

        sf = scaling_factor(tau, tau, xi, xi, L_decay) 
        common_factor_base = 1 / (k**2*norm_denom)
        common_factor = mu**2*common_factor_base 
        
        # base_etc_factor = mu**2 / (k**2 * norm_denom_etc) if k!=0 and norm_denom_etc>0 else 0.0
        base_etc_factor=common_factor
        if not np.isfinite(base_etc_factor): return np.zeros(5)
        cosx=np.cos(x); sinx=np.sin(x); six=sine_integral(x)
        if abs(x) < 1e-12: sinx_over_x = 1.0
        else: sinx_over_x = sinx / x

        term00_num=2.0*alpha**2*(-1.0+cosx+x*six);
        uetc_val[0]=term00_num * base_etc_factor
        if x==0 or alpha==0: termS=0.0
        else:
            term1_s=(8*(-18+x**2)+8*(-2+alpha**2)*v**2*(-18+x**2)+v**4*(8*(-18+x**2)-8*alpha**2*(-18+x**2)+alpha**4*(-54+11*x**2)))*cosx
            term2_s_num=(-32*(1+(-2+alpha**2)*v**2+(1-alpha**2+alpha**4)*v**4)*x**3+3*(-8*(-6+x**2)-8*(-2+alpha**2)*v**2*(-6+x**2)+v**4*(-8*(-6+x**2)+8*alpha**2*(-6+x**2)+alpha**4*(18+x**2)))*sinx)
            if abs(x) < 1e-12:
                term2_s = 0.0 
            else: term2_s=term2_s_num / x
            term3_s=(8+8*(-2+alpha**2)*v**2+(8-8*alpha**2+11*alpha**4)*v**4)*x**3*six
            termS_num=term1_s+term2_s+term3_s
            termS=termS_num/(16.0*alpha**2*x**2) if (x!=0 and alpha!=0) else 0.0
        uetc_val[1]=termS * base_etc_factor
        if x==0 or alpha==0: termV=0.0
        else:
             if abs(x) < 1e-12: 
                 tV1_sub = 0.0 
             else: tV1_sub = (x**3 + 3.0*x*cosx - 3.0*sinx) / (3.0 * x**3)
             term1_v=(2.0*(8.0+8.0*(-2.0+alpha**2)*v**2+(8.0-8.0*alpha**2+3*alpha**4)*v**4))*tV1_sub
             term2_v=alpha**4*v**4*(-2.0+cosx+sinx_over_x+x*six)
             termV_num=term1_v+term2_v; termV=termV_num/(8.0*alpha**2) if alpha!=0 else 0.0
        uetc_val[2]=termV * base_etc_factor
        if x==0 or alpha==0: termT=0.0
        else:
            term1_t=3*(8+8*(-2+alpha**2)*v**2+(8-8*alpha**2+3*alpha**4)*v**4)*(-2+x**2)*cosx
            term2_t_num=(64*(-1+v**2)*(1+(-1+alpha**2)*v**2)*x**3-3*(-8*(2+x**2)-8*(-2+alpha**2)*v**2*(2+x**2)+v**4*(-8*(2+x**2)+8*alpha**2*(2+x**2)+alpha**4*(-6+5*x**2)))*sinx)
            if abs(x) < 1e-12:
                term2_t = 0.0 
            else: term2_t = term2_t_num / x
            term3_t=3*(8+8*(-2+alpha**2)*v**2+(8-8*alpha**2+3*alpha**4)*v**4)*x**3*six
            termT_num=term1_t+term2_t+term3_t
            termT=termT_num/(96.0*alpha**2*x**2) if (x!=0 and alpha!=0) else 0.0
        uetc_val[3]=termT * base_etc_factor
        if x==0: 
            term00S_num=0.0
        else: term00S_num=(mu**2*(2+(-2+alpha**2)*v**2)*(-4+cosx+3*sinx_over_x+x*six))
        term00S=term00S_num/(2.*k**2*norm_denom_etc) if norm_denom_etc>1e-12 else 0.0
        uetc_val[4]=term00S
        uetc_val *= sf; return np.nan_to_num(uetc_val)

    # --- Regime 3: General Case ---
    if uetc_feedback > 1: print('General')
    n_terms_raw = max(min_terms, int(scale_terms * xp))
    n_terms = min(n_terms_raw, MAX_N_TERMS)
    use_approx = abs(x1 - x2) >= xapr
    small_rho = rho < 1e-2
    if use_approx: I1=I1_int_a(min(x1,x2),rho_safe); I4=I4_int_a(min(x1,x2),rho_safe)
    else: I1=I1_int(xm,rho_safe,n_terms)-I1_int(xp,rho_safe,n_terms); I4=I4_int(xm,rho_safe,n_terms)-I4_int(xp,rho_safe,n_terms);
    if not use_approx and small_rho: I4 = I1 / 2.0 
    I2=I2_int(xm,rho_safe)-I2_int(xp,rho_safe); I3=I3_int(xm,rho_safe)-I3_int(xp,rho_safe)
    if not use_approx and small_rho: I5=I2/2.0; I6=I3/2.0
    else: I5=I5_int(xm,rho_safe)-I5_int(xp,rho_safe); I6=I6_int(xm,rho_safe)-I6_int(xp,rho_safe)
    integrals = [I1, I2, I3, I4, I5, I6];
    if not all(np.isfinite(i) for i in integrals): return np.zeros(5)
    safe_a1a2rho2=max(2.*alpha1*alpha2*rho_safe**2,1e-30); safe_a1a2=max(2.*alpha1*alpha2,1e-30)
    
    sum00=2*alpha1*alpha2*I1; uetc_val[0]=sum00*common_factor
    c_term1=(-27*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(1+(-1+2*alpha1**2)*v1**2)*(1+(-1+2*alpha2**2)*v2**2))/safe_a1a2rho2
    c_term2=(-3*(-9*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(-1+v2**2+v1**2*(1+(-1+(alpha1*alpha2)**2)*v2**2))))/safe_a1a2rho2
    c_term3=(-9*(1+(-1+alpha1**2)*v1**2)*(1+(-1+alpha2**2)*v2**2))/safe_a1a2
    c_term4=(-3*(-(alpha2**2*rho_safe**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(-18*alpha2**2*v2**2+rho_safe**2*(1+(-1+4*alpha2**2)*v2**2))))/safe_a1a2rho2
    c_term5=(3*(-(alpha2**2*rho_safe**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(-18*alpha2**2*v2**2+rho_safe**2*(1+(-1+4*alpha2**2)*v2**2))))/safe_a1a2rho2
    c_term6=(9*(-(alpha2**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(1+(-1+2*alpha2**2)*v2**2)))/safe_a1a2
    sumS=c_term1*I1+c_term2*I2+c_term3*I3+c_term4*I4+c_term5*I5+c_term6*I6; uetc_val[1]=sumS*common_factor
    safe_rho2_local=max(rho_safe**2,1e-30); safe_a1a2_local=max(alpha1*alpha2,1e-30)
    c_term1=(3*alpha1*alpha2*v1**2*v2**2)/safe_rho2_local; c_term2=(-3*alpha1*alpha2*v1**2*v2**2)/safe_rho2_local
    c_term3=((1+(-1+alpha1**2)*v1**2)*(1+(-1+alpha2**2)*v2**2))/safe_a1a2_local
    c_term4=(alpha1*alpha2*(-6+rho_safe**2)*v1**2*v2**2)/safe_rho2_local; c_term5=-((alpha1*alpha2*(-6+rho_safe**2)*v1**2*v2**2)/safe_rho2_local)
    c_term6=(alpha2**2*(-1+v1**2)*v2**2-alpha1**2*v1**2*(1+(-1+2*alpha2**2)*v2**2))/safe_a1a2_local
    sumV=c_term1*I1+c_term2*I2+c_term3*I3+c_term4*I4+c_term5*I5+c_term6*I6; uetc_val[2]=sumV*common_factor
    safe_4a1a2rho2=max(4.0*alpha1*alpha2*rho_safe**2,1e-30); safe_4a1a2=max(4.0*alpha1*alpha2,1e-30)
    c_term1=(-3.0*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(-1.0+v1**2)*(-1.0+v2**2))/safe_4a1a2rho2
    c_term2=(3.0*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(-1.0+v2**2+v1**2*(1.0+(-1.0+(alpha1*alpha2)**2)*v2**2)))/safe_4a1a2rho2
    c_term3=-((1.0+(-1.0+alpha1**2)*v1**2)*(1.0+(-1.0+alpha2**2)*v2**2))/safe_4a1a2
    c_term4=(-(alpha2**2*rho_safe**2*(-1.0+v1**2)*v2**2)+alpha1**2*v1**2*(6.0*alpha2**2*v2**2-rho_safe**2*(-1.0+v2**2)))/safe_4a1a2rho2
    c_term5=(alpha2**2*rho_safe**2*(-1.0+v1**2)*v2**2+alpha1**2*v1**2*(-6.0*alpha2**2*v2**2+rho_safe**2*(-1.0+v2**2)))/safe_4a1a2rho2
    c_term6=(-(alpha2**2*(-1.0+v1**2)*v2**2)+alpha1**2*v1**2*(1.0+(-1.0+2.0*alpha2**2)*v2**2))/safe_4a1a2
    sumT=c_term1*I1+c_term2*I2+c_term3*I3+c_term4*I4+c_term5*I5+c_term6*I6; uetc_val[3]=sumT*common_factor
    c_term1=(-(alpha2**2*(-1+v1**2))+alpha1**2*(1-v2**2+2*alpha2**2*(v1**2+v2**2)))/safe_a1a2
    c_term2=(-3*(-(alpha2**2*(-1+v1**2))+alpha1**2*(1-v2**2+alpha2**2*(v1**2+v2**2))))/safe_a1a2
    c_term3=0.0; c_term4=(-3*alpha1*alpha2*(v1**2+v2**2))/2.0; c_term5=(3*alpha1*alpha2*(v1**2+v2**2))/2.0; c_term6=0.0
    sumC=c_term1*I1+c_term2*I2+c_term3*I3+c_term4*I4+c_term5*I5+c_term6*I6; uetc_val[4]=sumC*common_factor
    return np.nan_to_num(uetc_val)

def calculate_uetc_cell_serial(i, j, ntau_local, tau_vals_local, k_value_local, spr_base_local):
    if j < i: return i, j, None # Lower triangle, will be filled by symmetry
    tau1 = tau_vals_local[i]
    tau2 = tau_vals_local[j]

    if tau1 <= 0 or tau2 <= 0 or k_value_local <= 0:
        uetc_results_raw = np.zeros(5)
    else:
        uetc_results_raw = get_uetc(tau1, tau2, k_value_local, spr_base_local)

    return i, j, uetc_results_raw


def diagonalize_uetc_matrices_minimal(uetc_matrices_raw, tau_vals_local, weighting_local, nmodes_local, k_current_val):
    n = len(tau_vals_local)
    if nmodes_local > n: nmodes_local = n

    # These are the raw UETC matrices <Θ(k,τᵢ)Θ(k,τⱼ)>_k from get_uetc via calculate_uetc_cell_serial
    ss00array_raw = uetc_matrices_raw['UETC_00']
    ssarray_raw   = uetc_matrices_raw['UETC_S']
    vvarray_raw   = uetc_matrices_raw['UETC_V']
    ttarray_raw   = uetc_matrices_raw['UETC_T']
    sscrossarray_raw = uetc_matrices_raw['UETC_00S']

    # Construct the (k²τᵢτⱼ)^γ weighting matrix for eq. (32)
    tau_i, tau_j = np.meshgrid(tau_vals_local, tau_vals_local, indexing='ij')

    base_for_power = (k_current_val**2) * tau_i * tau_j
    

    full_diagonalization_weight = np.power(base_for_power, weighting_local)
    full_diagonalization_weight = np.nan_to_num(full_diagonalization_weight)*(tau_i*tau_j)**0.5# For safety
    # full_diagonalization_weight= 1

    # Apply this full weighting to the raw UETC matrices to form the matrix to be diagonalized
    vv_matrix_weighted = full_diagonalization_weight * vvarray_raw 
    tt_matrix_weighted = full_diagonalization_weight * ttarray_raw 
    
    scalar_matrix_weighted = np.zeros((2 * n, 2 * n), dtype=float)
    scalar_matrix_weighted[0:n, 0:n]     = full_diagonalization_weight * ss00array_raw
    scalar_matrix_weighted[n:2*n, n:2*n] = full_diagonalization_weight * ssarray_raw
    scalar_matrix_weighted[0:n, n:2*n]   = full_diagonalization_weight * sscrossarray_raw
    scalar_matrix_weighted[n:2*n, 0:n]   = scalar_matrix_weighted[0:n, n:2*n].T # Symmetrize

    eigen_results = {}
    try:
        scalar_eval, scalar_evec = scipy.linalg.eigh(scalar_matrix_weighted)
        vv_eval, vv_evec           = scipy.linalg.eigh(vv_matrix_weighted)
        tt_eval, tt_evec           = scipy.linalg.eigh(tt_matrix_weighted)

        # Eigenvalues (λ_i(k)) are now from the correctly scaled matrix
        eigen_results['eval_S'] = scalar_eval[2*n - nmodes_local:][::-1]
        top_scalar_evecs        = scalar_evec[:, 2*n - nmodes_local:][:, ::-1]
        # Eigenvectors (u_i(k,τ))
        eigen_results['evec_00']= top_scalar_evecs[0:n, :].T 
        eigen_results['evec_S'] = top_scalar_evecs[n:2*n, :].T

        eigen_results['eval_V'] = vv_eval[n - nmodes_local:][::-1]
        eigen_results['evec_V'] = vv_evec[:, n - nmodes_local:][:, ::-1].T

        eigen_results['eval_T'] = tt_eval[n - nmodes_local:][::-1]
        eigen_results['evec_T'] = tt_evec[:, n - nmodes_local:][:, ::-1].T
        
        
        eigen_results['eval_00']=scalar_eval[n - nmodes_local:n][::-1]

    except np.linalg.LinAlgError as e:
        print(f"ERROR during diagonalization for k={k_current_val:.4e}: {e}")
        empty_evec = np.full((nmodes_local, n), np.nan)
        empty_eval = np.full(nmodes_local, np.nan)
        eigen_results['eval_S'] = empty_eval; eigen_results['evec_00'] = empty_evec
        eigen_results['evec_S'] = empty_evec; eigen_results['eval_V'] = empty_eval
        eigen_results['evec_V'] = empty_evec; eigen_results['eval_T'] = empty_eval
        eigen_results['evec_T'] = empty_evec
        eigen_results['error'] = str(e)
    return eigen_results

def calculate_normalized_uetc_eigenvectors(k_val, spr_parameters, tau_vals_local, weighting_local, nmodes_local, normalize_eigenvectors=True):
    ntau_calc = len(tau_vals_local)

    uetc_matrices_raw = {key: np.zeros((ntau_calc, ntau_calc)) for key in ['UETC_00','UETC_S','UETC_V','UETC_T','UETC_00S']}

    for i in range(ntau_calc):
        for j in range(i, ntau_calc): # Iterate only for upper triangle + diagonal
            _, _, raw_uetc_results = calculate_uetc_cell_serial( # MODIFIED: was scaled_results
                i, j, ntau_calc, tau_vals_local, k_val, spr_parameters
            )
            if raw_uetc_results is not None:
                uetc_matrices_raw['UETC_00'][i,j] = raw_uetc_results[0]
                uetc_matrices_raw['UETC_S'][i,j]  = raw_uetc_results[1]
                uetc_matrices_raw['UETC_V'][i,j]  = raw_uetc_results[2]
                uetc_matrices_raw['UETC_T'][i,j]  = raw_uetc_results[3]
                uetc_matrices_raw['UETC_00S'][i,j]= raw_uetc_results[4]
                if i != j: # Symmetrize
                    uetc_matrices_raw['UETC_00'][j,i] = raw_uetc_results[0]
                    uetc_matrices_raw['UETC_S'][j,i]  = raw_uetc_results[1]
                    uetc_matrices_raw['UETC_V'][j,i]  = raw_uetc_results[2]
                    uetc_matrices_raw['UETC_T'][j,i]  = raw_uetc_results[3]
                    uetc_matrices_raw['UETC_00S'][j,i]= raw_uetc_results[4] # Cross-term is also symmetric if Θ₀₀Θₛ = ΘₛΘ₀₀
    
    # Diagonalization is now performed on (k²τᵢτⱼ)^γ * <ΘΘ>
    diag_results = diagonalize_uetc_matrices_minimal(
        uetc_matrices_raw, tau_vals_local, weighting_local, nmodes_local, k_val
    )



    if 'error' in diag_results:
        return { 
            'k_value': k_val, 'tau_values': tau_vals_local,
            'eigenvectors_00': np.full((nmodes_local, ntau_calc), np.nan),
            'eigenvectors_S': np.full((nmodes_local, ntau_calc), np.nan),
            'eigenvectors_V': np.full((nmodes_local, ntau_calc), np.nan),
            'eigenvectors_T': np.full((nmodes_local, ntau_calc), np.nan),
            'eigenvalues_S': np.full(nmodes_local, np.nan),
            'eigenvalues_V': np.full(nmodes_local, np.nan),
            'eigenvalues_T': np.full(nmodes_local, np.nan),
            'error': diag_results['error']
        }

    output = {
        'k_value': k_val,
        'tau_values': tau_vals_local,
        'eigenvectors_00': diag_results['evec_00'],
        'eigenvectors_S': diag_results['evec_S'],
        'eigenvectors_V': diag_results['evec_V'],
        'eigenvectors_T': diag_results['evec_T'],
        'eigenvalues_S': diag_results['eval_S'],
        'eigenvalues_00': diag_results['eval_00'],
        'eigenvalues_V': diag_results['eval_V'],
        'eigenvalues_T': diag_results['eval_T'],
    }
    
    # print(diag_results['eval_00'])
    # print(diag_results['eval_S'])

    # if normalize_eigenvectors:
    #     for key in ['eigenvectors_00', 'eigenvectors_S', 'eigenvectors_V', 'eigenvectors_T']:
    #         if output[key] is not None and output[key].size > 0 and not np.all(np.isnan(output[key])):
    #             for i_mode in range(output[key].shape[0]):
    #                 vec = output[key][i_mode, :]
    #                 if not np.all(np.isnan(vec)):
    #                     norm = np.linalg.norm(vec[~np.isnan(vec)])
    #                     if norm > 1e-12: # Avoid division by zero for null eigenvectors
    #                         output[key][i_mode, :] = vec / norm
    return output,uetc_matrices_raw

def worker_calculate_for_k(args_tuple):
    k_val, spr_params, tau_grid_worker, weighting_param, nmodes_param = args_tuple
    result_for_k,_ = calculate_normalized_uetc_eigenvectors(
        k_val, spr_params, tau_grid_worker, weighting_param, nmodes_param
    )
    return k_val, result_for_k


def plot_uetc_evecs_reconstruction(k_to_plot, tau_values, string_params_obj,
                                   weighting_gamma_param, nmodes_total_calculated,
                                   num_evecs_to_plot=10, num_modes_for_reconstruction=10,
                                   uetc_n_levels=15):
    """
    Generates plots for:
    1. Original Scaled UETC components.
    2. Top N eigenvectors.
    3. UETC components reconstructed from top M eigenvectors.
    UETC Scaling for display: (τ₁τ₂)¹/² <Θ(k, τ₁)Θ(k, τ₂)> / μ².
    """
    print(f"--- Generating UETC, E-vec, & Reconstruction plots for k = {k_to_plot:.4e} ---")

    if k_to_plot <= 0: print("Error: k_to_plot must be positive."); return
    valid_tau_mask = tau_values > 0
    if not np.any(valid_tau_mask): print("Error: No positive tau values."); return
    
    ntau = len(tau_values)
    if ntau < 2: print("Error: Need at least 2 tau points."); return

    log_kt_axis = np.log10(k_to_plot * tau_values)
    mu_sq = string_params_obj.mu**2

    # Ensure we don't try to plot/use more modes than calculated
    actual_num_evecs_to_plot = min(num_evecs_to_plot, nmodes_total_calculated)
    actual_modes_for_reconstruction = min(num_modes_for_reconstruction, nmodes_total_calculated)
    
    if num_modes_for_reconstruction > nmodes_total_calculated:
        print(f"Warning: Requested {num_modes_for_reconstruction} for reconstruction, but only {nmodes_total_calculated} available. Using {nmodes_total_calculated}.")

    # --- Calculate UETC matrices and Eigenvectors ---
    eigen_data, uetc_raw_matrices = calculate_normalized_uetc_eigenvectors(
        k_to_plot, string_params_obj, tau_values,
        weighting_gamma_param, nmodes_total_calculated,
        normalize_eigenvectors=False # Normalization doesn't affect reconstruction if λ*u*u used
    )

    if eigen_data is None or uetc_raw_matrices is None or 'error' in eigen_data:
        err_msg = eigen_data.get('error', "Unknown error") if eigen_data else "Unknown error"
        print(f"Error: Failed to calculate UETC/Eigenvectors for k={k_to_plot}. Details: {err_msg}")
        # Create a dummy figure to show the error
        fig, ax = plt.subplots(1,1)
        ax.text(0.5, 0.5, f"Failed to get data for k={k_to_plot}\n{err_msg}", ha='center', va='center')
        plt.show()
        return

    # --- 1. Prepare Original UETC data for plotting (apply display scaling) ---
    uetc_plot_data_scaled = [np.full((ntau, ntau), np.nan) for _ in range(5)]
    raw_matrix_keys = ['UETC_00', 'UETC_S', 'UETC_V', 'UETC_T', 'UETC_00S']
    for comp_idx, matrix_key in enumerate(raw_matrix_keys):
        raw_matrix = uetc_raw_matrices[matrix_key] # This is <ΘΘ>, includes (Gμ)² and model's sf
        for i in range(ntau): # tau1
            tau1 = tau_values[i]
            for j in range(ntau): # tau2
                tau2 = tau_values[j]
                raw_value = raw_matrix[j, i] # raw_matrix is [tau2_idx, tau1_idx]
                plot_display_scaling = (tau1 * tau2)**0.5 / mu_sq if mu_sq != 0 else 0
                # plot_display_scaling = 1 if mu_sq != 0 else 0

                uetc_plot_data_scaled[comp_idx][j, i] = raw_value * plot_display_scaling

    # --- 3. Reconstruct UETCs from eigenvectors and eigenvalues ---
    # M_reconstructed(τᵢ,τⱼ) = ( Σ_p λ_p u_p(τᵢ)u_p(τⱼ) ) / W(τᵢ,τⱼ)
    # W(τᵢ,τⱼ) = (k²τᵢτⱼ)^γ * (τᵢτⱼ)^0.5
    
    uetc_reconstructed_plot_data = [np.full((ntau, ntau), np.nan) for _ in range(5)]

    tau_i_mesh, tau_j_mesh = np.meshgrid(tau_values, tau_values, indexing='ij') # For W calculation
    
    # Calculate the W_ij = (k² τᵢ τⱼ)^γ * (τᵢ τⱼ)^0.5 factor for un-weighting
    # Ensure base for power is positive
    base_for_W_power = (k_to_plot**2) * tau_i_mesh * tau_j_mesh
    # Handle potential zeros or negatives if tau_values aren't strictly positive (though logspace should be)
    base_for_W_power[base_for_W_power <= 0] = 1e-100 # Avoid log(0) or power of negative
    
    W_ij_unweight_factor = np.power(base_for_W_power, weighting_gamma_param) * np.sqrt(tau_i_mesh * tau_j_mesh)
    W_ij_unweight_factor[W_ij_unweight_factor == 0] = 1e-100 # Avoid division by zero

    # Scalar components (00, S, 00S) from combined diagonalization
    if eigen_data['eigenvalues_S'] is not None and \
       eigen_data['eigenvectors_00'] is not None and \
       eigen_data['eigenvectors_S'] is not None:
        
        lambda_S_modes = eigen_data['eigenvalues_S'][:actual_modes_for_reconstruction]
        lambda_00_modes = eigen_data['eigenvalues_00'][:actual_modes_for_reconstruction]

        u_00_modes = eigen_data['eigenvectors_00'][:actual_modes_for_reconstruction, :] # (n_reco_modes, ntau)
        u_S_modes = eigen_data['eigenvectors_S'][:actual_modes_for_reconstruction, :]   # (n_reco_modes, ntau)

        # Reconstruct M_00_raw, M_S_raw, M_00S_raw
        # Sum over p: λ_p * u_p(τ_i) * u_p(τ_j)
        # For M_00: sum_p λ_p_S * u_p_00(τ_i) * u_p_00(τ_j)
        # Using einsum for efficiency: 'p,pi,pj->ij' where p is mode, i is tau1_idx, j is tau2_idx
        M_00_weighted_sum = np.einsum('p,pi,pj->ij', lambda_S_modes, u_00_modes, u_00_modes)
        M_S_weighted_sum  = np.einsum('p,pi,pj->ij', lambda_S_modes, u_S_modes, u_S_modes)
        M_00S_weighted_sum= np.einsum('p,pi,pj->ij', lambda_S_modes, u_00_modes, u_S_modes) # u_00(τi) u_S(τj)

        M_00_reco_raw = M_00_weighted_sum / W_ij_unweight_factor.T # W_ij is ij, sum is ij (tau1, tau2)
        M_S_reco_raw  = M_S_weighted_sum  / W_ij_unweight_factor.T
        M_00S_reco_raw= M_00S_weighted_sum/ W_ij_unweight_factor.T
        
        # Apply display scaling
        plot_display_scaling_mesh = np.sqrt(tau_i_mesh * tau_j_mesh) / mu_sq if mu_sq != 0 else np.zeros_like(tau_i_mesh)
        
        uetc_reconstructed_plot_data[0] = (M_00_reco_raw * plot_display_scaling_mesh).T
        uetc_reconstructed_plot_data[1] = (M_S_reco_raw  * plot_display_scaling_mesh).T
        uetc_reconstructed_plot_data[4] = (M_00S_reco_raw* plot_display_scaling_mesh).T

    # Vector component
    if eigen_data['eigenvalues_V'] is not None and eigen_data['eigenvectors_V'] is not None:
        lambda_V_modes = eigen_data['eigenvalues_V'][:actual_modes_for_reconstruction]
        u_V_modes = eigen_data['eigenvectors_V'][:actual_modes_for_reconstruction, :]
        M_V_weighted_sum = np.einsum('p,pi,pj->ij', lambda_V_modes, u_V_modes, u_V_modes)
        M_V_reco_raw = M_V_weighted_sum / W_ij_unweight_factor
        # uetc_reconstructed_plot_data[2] = M_V_reco_raw.T * plot_display_scaling_mesh
        uetc_reconstructed_plot_data[2] = (M_V_reco_raw * plot_display_scaling_mesh).T

    # Tensor component
    if eigen_data['eigenvalues_T'] is not None and eigen_data['eigenvectors_T'] is not None:
        lambda_T_modes = eigen_data['eigenvalues_T'][:actual_modes_for_reconstruction]
        u_T_modes = eigen_data['eigenvectors_T'][:actual_modes_for_reconstruction, :]
        M_T_weighted_sum = np.einsum('p,pi,pj->ij', lambda_T_modes, u_T_modes, u_T_modes)
        M_T_reco_raw = M_T_weighted_sum / W_ij_unweight_factor
        # uetc_reconstructed_plot_data[3] = M_T_reco_raw.T * plot_display_scaling_mesh
        uetc_reconstructed_plot_data[3] = (M_T_reco_raw * plot_display_scaling_mesh).T



    # --- Plotting ---
    fig, axes = plt.subplots(5, 3, figsize=(18, 24), constrained_layout=True)
    
    component_uetc_titles_orig = [
        r"Orig. Scaled $\langle \Theta_{00}\Theta_{00} \rangle$",
        r"Orig. Scaled $\langle \Theta_S\Theta_S \rangle$",
        r"Orig. Scaled $\langle \Theta_V\Theta_V \rangle$",
        r"Orig. Scaled $\langle \Theta_T\Theta_T \rangle$",
        r"Orig. Scaled $\langle \Theta_{00}\Theta_S \rangle$"
    ]
    eigenvector_keys = ['eigenvectors_00', 'eigenvectors_S', 'eigenvectors_V', 'eigenvectors_T', None]
    eigenvector_titles_suffix = ["(00 Type)", "(S Type)", "(V Type)", "(T Type)", "(Scalar Modes)"]
    component_uetc_titles_reco = [
        r"Reco. Scaled $\langle \Theta_{00}\Theta_{00} \rangle$",
        r"Reco. Scaled $\langle \Theta_S\Theta_S \rangle$",
        r"Reco. Scaled $\langle \Theta_V\Theta_V \rangle$",
        r"Reco. Scaled $\langle \Theta_T\Theta_T \rangle$",
        r"Reco. Scaled $\langle \Theta_{00}\Theta_S \rangle$"
    ]

    for comp_idx in range(5):
        ax_orig_uetc = axes[comp_idx, 0]
        ax_evec = axes[comp_idx, 1]
        ax_reco_uetc = axes[comp_idx, 2]

        # Plot Original UETC (Col 0)
        data_plot_orig = np.ma.masked_invalid(uetc_plot_data_scaled[comp_idx])
        if not data_plot_orig.mask.all():
            vmin, vmax = np.nanmin(data_plot_orig), np.nanmax(data_plot_orig)
            if vmin == vmax: vmin -= abs(vmin*.1) if vmin!=0 else .1; vmax += abs(vmax*.1) if vmax!=0 else .1
            if vmin == vmax: vmin, vmax = vmin-0.1, vmax+0.1 # ensure range
            levels = np.linspace(vmin, vmax, uetc_n_levels) if vmin < vmax else uetc_n_levels
            contour = ax_orig_uetc.contourf(log_kt_axis, log_kt_axis, data_plot_orig,
                                           levels=levels, cmap='jet', vmin=vmin, vmax=vmax, extend='both')
            fig.colorbar(contour, ax=ax_orig_uetc, orientation='vertical', fraction=0.046, pad=0.04)
        ax_orig_uetc.set_title(component_uetc_titles_orig[comp_idx], fontsize=9)
        ax_orig_uetc.set_xlabel(r"$\log_{10}(k \tau_1)$"); ax_orig_uetc.set_ylabel(r"$\log_{10}(k \tau_2)$")
        ax_orig_uetc.set_aspect('equal', adjustable='box')

        # Plot Eigenvectors (Col 1)
        evec_data_key = eigenvector_keys[comp_idx]
        evec_title_suffix_val = eigenvector_titles_suffix[comp_idx]
        if comp_idx == 4: # 00S cross-term: e-vecs are scalar
             ax_evec.text(0.5, 0.5, "Scalar Eigenvectors\n(Rows 1 & 2)", ha='center', va='center', transform=ax_evec.transAxes)
        elif evec_data_key and eigen_data[evec_data_key] is not None:
            current_evecs = eigen_data[evec_data_key]
            if not np.all(np.isnan(current_evecs)):
                for i_mode in range(min(actual_num_evecs_to_plot, current_evecs.shape[0])):
                    ax_evec.plot(log_kt_axis, current_evecs[i_mode, :], label=f"M {i_mode+1}")
                if actual_num_evecs_to_plot > 0 and current_evecs.shape[0] > 0: ax_evec.legend(fontsize='xx-small', loc='best')
            else: ax_evec.text(0.5, 0.5, "NaN E-vecs", ha='center', va='center')
        else: ax_evec.text(0.5, 0.5, "No E-vec Data", ha='center', va='center')
        ax_evec.set_title(f"Top E-vecs {evec_title_suffix_val}", fontsize=9)
        ax_evec.set_xlabel(r"$\log_{10}(k \tau)$"); ax_evec.set_ylabel(r"$u_i(k,\tau)$")
        ax_evec.grid(True, linestyle=':', alpha=0.7)

        # Plot Reconstructed UETC (Col 2)
        data_plot_reco = np.ma.masked_invalid(uetc_reconstructed_plot_data[comp_idx])
        if not data_plot_reco.mask.all():
            vmin, vmax = np.nanmin(data_plot_reco), np.nanmax(data_plot_reco)
            if vmin == vmax: vmin -= abs(vmin*.1) if vmin!=0 else .1; vmax += abs(vmax*.1) if vmax!=0 else .1
            if vmin == vmax: vmin, vmax = vmin-0.1, vmax+0.1 # ensure range
            levels = np.linspace(vmin, vmax, uetc_n_levels) if vmin < vmax else uetc_n_levels
            contour = ax_reco_uetc.contourf(log_kt_axis, log_kt_axis, data_plot_reco,
                                           levels=levels, cmap='jet', vmin=vmin, vmax=vmax, extend='both')
            fig.colorbar(contour, ax=ax_reco_uetc, orientation='vertical', fraction=0.046, pad=0.04)
        ax_reco_uetc.set_title(component_uetc_titles_reco[comp_idx], fontsize=9)
        ax_reco_uetc.set_xlabel(r"$\log_{10}(k \tau_1)$"); ax_reco_uetc.set_ylabel(r"$\log_{10}(k \tau_2)$")
        ax_reco_uetc.set_aspect('equal', adjustable='box')

    fig.suptitle(f"UETC Analysis (k={k_to_plot:.3e}, Reco. with {actual_modes_for_reconstruction} modes)", fontsize=14)
    # plt.savefig(f"uetc_evec_reco_plots_k_{k_to_plot:.2e}.png", dpi=150)
    plt.show()
    print(f"--- Finished UETC, E-vec, & Reconstruction plots ---")

def align_eigenvector_signs(eigenfunctions_array, eigenfunctions_deriv_array):
    """
    Align the signs of eigenvectors across k values to ensure smooth evolution.
    
    Parameters:
    -----------
    eigenfunctions_array : np.ndarray
        Shape (nk, num_eigen_types, nmodes, nktau) - the eigenvectors u_i(k,τ)
    eigenfunctions_deriv_array : np.ndarray  
        Shape (nk, num_eigen_types, nmodes, nktau) - the derivatives du_i/d(log(kτ))
    
    Returns:
    --------
    Tuple of (aligned_eigenfunctions, aligned_derivatives)
    """
    print("Aligning eigenvector signs across k values...")
    
    # Make copies to avoid modifying original arrays
    aligned_eigenfunctions = eigenfunctions_array.copy()
    aligned_derivatives = eigenfunctions_deriv_array.copy()
    
    nk, num_eigen_types, nmodes, nktau = eigenfunctions_array.shape
    
    # Process each eigenfunction type separately
    for type_idx in range(num_eigen_types):
        print(f"  Processing eigenfunction type {type_idx}...")
        
        # For each mode within this type
        for mode_idx in range(nmodes):
            v_prev = None
            
            # Loop through k values
            for k_idx in range(nk):
                # Current eigenvector for this (type, mode, k)
                current_vec = aligned_eigenfunctions[k_idx, type_idx, mode_idx, :]
                
                # Skip if current vector is all NaN
                if np.all(np.isnan(current_vec)):
                    continue
                
                # If we have a previous vector, align signs
                if v_prev is not None and not np.all(np.isnan(v_prev)):
                    # Calculate overlap (dot product) between current and previous
                    # Only use valid (non-NaN) points for both vectors
                    valid_mask = ~(np.isnan(current_vec) | np.isnan(v_prev))
                    
                    if np.sum(valid_mask) > 0:  # Need at least some valid points
                        dot_product = np.sum(current_vec[valid_mask] * v_prev[valid_mask])
                        
                        # If overlap is negative, flip the sign
                        if dot_product < 0:
                            aligned_eigenfunctions[k_idx, type_idx, mode_idx, :] *= -1
                            aligned_derivatives[k_idx, type_idx, mode_idx, :] *= -1
                            # Update current_vec for next iteration (get fresh reference after modification)
                            current_vec = aligned_eigenfunctions[k_idx, type_idx, mode_idx, :]
                
                # Store current vector as previous for next k
                v_prev = current_vec.copy()
    
    print("Sign alignment completed.")
    return aligned_eigenfunctions, aligned_derivatives

# ======================================================
# Main Script Logic for Table Generation
# ======================================================

if __name__ == "__main__":

    # print(get_uetc(1, 1, 10**-3, SPRa))
    # print(get_uetc(10**-3, 10**-3, 1, SPRa))

    
    k_value_for_plotting=10**-3;
    tau_grid_plot = np.logspace(np.log10(10**-2/k_value_for_plotting),np.log10(10**2.5/k_value_for_plotting), 256)

    plot_uetc_evecs_reconstruction(
             k_value_for_plotting,
             tau_grid_plot,
             SPRa,
             weighting_gamma_param=weighting,         # Global 'weighting' (gamma)
             nmodes_total_calculated=nmodes,          # Global 'nmodes'
             num_evecs_to_plot=min(5, nmodes),       # How many eigenvectors to show in middle column
             num_modes_for_reconstruction=200, # How many modes to use for UETC reconstruction
             uetc_n_levels=200
         )
        
    
    print("--- Starting UETC Eigenvector Table Generation (Corrected Scalings & Derivatives) ---") # Modified print
    overall_start_time = time.time()

    k_grid = np.logspace(np.log10(k_min_calc), np.log10(k_max_calc), nk_calc)
    ktau_grid = np.logspace(np.log10(ktau_min_calc), np.log10(ktau_max_calc), nktau_calc)


    #tau_grid = np.logspace(np.log10(tau_min_calc), np.log10(tau_max_calc), ntau_calc)

    num_eigen_types = 4 # 00, S, V, T
    # all_eigenfunctions stores u_i(k,τ)
    all_eigenfunctions = np.zeros((nk_calc, num_eigen_types, nmodes, nktau_calc), dtype=np.float64)
    # !JR ADDITION: Table for du_i/d(log(kτ))
    all_eigenfunctions_d_dlogkt = np.zeros((nk_calc, num_eigen_types, nmodes, nktau_calc), dtype=np.float64)

    all_eigenvalues_S = np.zeros((nk_calc, nmodes), dtype=np.float64)
    all_eigenvalues_00 = np.zeros((nk_calc, nmodes), dtype=np.float64)

    all_eigenvalues_V = np.zeros((nk_calc, nmodes), dtype=np.float64)
    all_eigenvalues_T = np.zeros((nk_calc, nmodes), dtype=np.float64)

    tasks_args = [(k_val, SPRa, ktau_grid / k_val, weighting, nmodes) for k_val in k_grid]
    
    num_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    print(f"Starting k-loop with {num_workers} workers for {nk_calc} k-values (ntau={nktau_calc} per k)...")

    k_to_idx_map = {k_val: idx for idx, k_val in enumerate(k_grid)}
    processed_k_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_k = {executor.submit(worker_calculate_for_k, args_tuple): args_tuple[0] for args_tuple in tasks_args}

        for future in concurrent.futures.as_completed(future_to_k):
            original_k_val = future_to_k[future]
            k_idx = k_to_idx_map[original_k_val]
            processed_k_count += 1
            try:
                _, data_for_k = future.result()
                
                if data_for_k and 'error' not in data_for_k:
                    # Store eigenfunctions as before
                    all_eigenfunctions[k_idx, 0, :, :] = data_for_k['eigenvectors_00']
                    all_eigenfunctions[k_idx, 1, :, :] = data_for_k['eigenvectors_S']
                    all_eigenfunctions[k_idx, 2, :, :] = data_for_k['eigenvectors_V']
                    all_eigenfunctions[k_idx, 3, :, :] = data_for_k['eigenvectors_T']
                    
                    all_eigenvalues_S[k_idx, :] = data_for_k['eigenvalues_S']
                    all_eigenvalues_00[k_idx, :] = data_for_k['eigenvalues_00']

                    all_eigenvalues_V[k_idx, :] = data_for_k['eigenvalues_V']
                    all_eigenvalues_T[k_idx, :] = data_for_k['eigenvalues_T']

                    # We only need derivatives for scalar components 00 and S for now
                    # based on the old code's emt00dot and emtSdot
                    current_k_val_for_deriv = k_grid[k_idx] # Use the actual k value from the grid
                    
                    # Ensure k_val is positive before taking log
                    if current_k_val_for_deriv <= 0:
                        print(f"Warning: k_val <= 0 ({current_k_val_for_deriv}) at k_idx={k_idx}, cannot compute log(kτ). Skipping derivative.")
                        # Derivatives will remain zero for this k_idx if this happens
                    else:
                        log_ktau_axis = np.log(ktau_grid)

                        for type_idx_deriv in range(2): # 0 for evec_00, 1 for evec_S
                            for mode_idx_deriv in range(nmodes):
                                eigenfunc_values_1d = all_eigenfunctions[k_idx, type_idx_deriv, mode_idx_deriv, :]
                                
                                if np.all(np.isnan(eigenfunc_values_1d)) or eigenfunc_values_1d.size < 2 : # Need at least 2 points for spline
                                    all_eigenfunctions_d_dlogkt[k_idx, type_idx_deriv, mode_idx_deriv, :] = np.nan
                                    # print(f"Skipping derivative for k_idx={k_idx}, type={type_idx_deriv}, mode={mode_idx_deriv} due to NaNs or insufficient points.")
                                    continue
                                try:
                                    # Mask NaNs for spline fitting if any partial NaNs exist
                                    valid_indices = ~np.isnan(eigenfunc_values_1d)
                                    if np.sum(valid_indices) < 2: # Need at least 2 valid points
                                         all_eigenfunctions_d_dlogkt[k_idx, type_idx_deriv, mode_idx_deriv, :] = np.nan
                                         continue
                                    
                                    spl = scipy.interpolate.CubicSpline(log_ktau_axis[valid_indices], eigenfunc_values_1d[valid_indices], extrapolate=False)
                                    deriv_vals_on_grid = spl.derivative(nu=1)(log_ktau_axis) # Evaluate on original log_ktau grid points
                                    
                                    # Handle extrapolation if needed, or fill with NaN (CubicSpline with extrapolate=False will NaN outside range)
                                    # For points where extrapolation occurred, or if original was NaN, deriv_vals_on_grid will be NaN.
                                    all_eigenfunctions_d_dlogkt[k_idx, type_idx_deriv, mode_idx_deriv, :] = deriv_vals_on_grid

                                except ValueError as ve:
                                    # This can happen if log_ktau_axis is not strictly increasing, or other spline issues.
                                    print(f"CubicSpline ValueError for k_idx={k_idx}, type={type_idx_deriv}, mode={mode_idx_deriv}: {ve}")
                                    all_eigenfunctions_d_dlogkt[k_idx, type_idx_deriv, mode_idx_deriv, :] = np.nan
                                except Exception as e_spline:
                                    print(f"Spline/Derivative ERROR for k_idx={k_idx}, type={type_idx_deriv}, mode={mode_idx_deriv}: {e_spline}")
                                    all_eigenfunctions_d_dlogkt[k_idx, type_idx_deriv, mode_idx_deriv, :] = np.nan
                        # Derivatives for V and T are set to 0 for now, as not immediately used based on old code analysis
                        all_eigenfunctions_d_dlogkt[k_idx, 2, :, :] = 0.0 # Type 2: Vector
                        all_eigenfunctions_d_dlogkt[k_idx, 3, :, :] = 0.0 # Type 3: Tensor

                elif data_for_k and 'error' in data_for_k:
                     print(f"Processed k = {original_k_val:.4e} (index {k_idx}) but DIAGONALIZATION FAILED: {data_for_k['error']}. {processed_k_count}/{nk_calc} done.")
                     # Ensure derivative tables are also NaN for this k_idx
                     all_eigenfunctions_d_dlogkt[k_idx, :, :, :] = np.nan
                else:
                     print(f"Processed k = {original_k_val:.4e} (index {k_idx}) but received None result. {processed_k_count}/{nk_calc} done.")
                     all_eigenfunctions_d_dlogkt[k_idx, :, :, :] = np.nan


            except Exception as exc:
                print(f"Task for k={original_k_val:.4e} (index {k_idx}) generated an EXCEPTION: {exc}. {processed_k_count}/{nk_calc} done.")
                all_eigenfunctions[k_idx, :, :, :] = np.nan # Ensure main table is also NaN
                all_eigenfunctions_d_dlogkt[k_idx, :, :, :] = np.nan
            
            if processed_k_count % max(1, nk_calc // 10) == 0 or processed_k_count == nk_calc:
                 print(f"Progress: {processed_k_count}/{nk_calc} k-values processed.")

    # Apply sign alignment to eigenfunctions and derivatives
    print("\n--- Applying Sign Alignment ---")
    all_eigenfunctions, all_eigenfunctions_d_dlogkt = align_eigenvector_signs(
        all_eigenfunctions, all_eigenfunctions_d_dlogkt
    )

    overall_end_time = time.time()
    print(f"\n--- Table Generation Finished ---")
    print(f"Total time: {overall_end_time - overall_start_time:.2f} seconds.")

    print("\nShapes of data to be saved:")
    print(f"k_grid shape: {k_grid.shape} (nk)")
    print(f"ktau_grid shape: {ktau_grid.shape} (nktau)")
    print(f"all_eigenfunctions shape: {all_eigenfunctions.shape} (nk, num_eigen_types, nmodes, nktau)")
    print(f"all_eigenfunctions_d_dlogkt shape: {all_eigenfunctions_d_dlogkt.shape}") # !JR ADDED print
    print(f"all_eigenvalues_S shape: {all_eigenvalues_S.shape} (nk, nmodes)")

    try:
        np.savez("uetc_eigen_data_corrected_with_derivs.npz", 
                 k_grid=k_grid,
                 ktau_grid=ktau_grid,
                 eigenfunctions=all_eigenfunctions, # u_i(k,τ)
                 eigenfunctions_d_dlogkt=all_eigenfunctions_d_dlogkt, #du_i/d(log(kτ))
                 eigenvalues_S=all_eigenvalues_S,   # λ_i(k) for Scalar modes
                 eigenvalues_00=all_eigenvalues_00,
                 eigenvalues_V=all_eigenvalues_V,   # λ_i(k) for Vector modes
                 eigenvalues_T=all_eigenvalues_T,   # λ_i(k) for Tensor modes
                 string_params_mu=SPRa.mu,
                 string_params_alpha=SPRa.alpha,
                 string_params_v=SPRa.v,
                 string_params_xi=SPRa.xi,
                 string_params_L=SPRa.L,
                 nmodes=nmodes,
                 weighting_gamma=weighting)
        print("\nSaved data to uetc_eigen_data_corrected_with_derivs.npz") 
    except Exception as e:
        print(f"\nError saving data to .npz file: {e}")
    
    
    