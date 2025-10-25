# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 23:36:22 2022

@author: Rafael
"""

import numpy as np 
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d 
from itertools import product 
from scipy.integrate import quad 
from scipy.fft import fht 
import math 
import time 
data=np.loadtxt('explanatory10_pk.dat', unpack=True) #Cambia el .dat respecto al modelo usado para calcular el power spectrum en CLASS.
column=data
splits=np.array_split(column,2)
for array in splits:
    list(array)
x=column[0,] 
y=column[1,] 
x_log=np.log(x)
y_log=np.log(y)
kj2, P0slist = np.loadtxt('P0smooth_z234.txt')

def straight(x_log, a2, b2):
    return (a2 * x_log) + b2
popt, _ = curve_fit(straight, x_log[:20], y_log[:20])
a2, b2 = popt
popt, _ = curve_fit(straight, x_log[-5:], y_log[-5:])
c2, d2 = popt
f=interp1d(x, y, kind='cubic', bounds_error=False)
def Plin(k):
    if k<1 and k>1.04205722e-05:
        interpo=np.float(f(k))
    elif k<=1.04205722e-05:
        logk=np.log(k)
        logy=straight(logk, a2, b2)
        interpo=np.exp(logy)
    else:
        logk=np.log(k)
        logy=straight(logk, c2, d2)
        interpo=np.exp(logy)
    return interpo
Plin=np.vectorize(Plin) 
h0=interp1d(kj2, P0slist, kind='cubic', bounds_error=False)
def Psmooth(kj2):
    return np.float(h0(kj2))

Psmooth=np.vectorize(Psmooth)
def Ppeak(k):
    return Plin(k)-Psmooth(k)
Ppeak=np.vectorize(Ppeak)

#Parameters for fast Hankel transformation
n=100
j=np.linspace(0, n, n) 
kmin_xi=1e-3 
kmax_xi=1 
kc_xi=np.sqrt(kmin_xi*kmax_xi) 
jc=n/2 
m_xi=math.log10(kmax_xi/kc_xi) 
dln_xi=(m_xi/(n-jc))*np.log(10) 
kj_xi=kc_xi*np.exp((j-jc)*dln_xi) 
rc_xi=1/kc_xi
rj_xi=rc_xi*np.exp((j-jc)*dln_xi)
bF_true=-0.131
beta_true=1.580
alphap_true=1.0
alphat_true=1.0
sigp=6.41 
sigt=3.26 
Apeak=1.0
kNL=6.77 
kP=15.9 
kv0=0.819 
kV=0.917 
aNL=0.550
aP=2.12
av=1.5
aV=0.528
Rp=4 

def argI0s(mk, kj, gm, av, Rp):
        return np.exp(-gm**av*mk**av)*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2
def argI1s(mk, kj, gm, av, Rp):
        return np.exp(-gm**av*mk**av)*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2*mk**2
def argI2s(mk, kj, gm, av, Rp):
        return np.exp(-gm**av*mk**av)*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2*mk**4
def argI3s(mk, kj, gm, av, Rp):
        return np.exp(-gm**av*mk**av)*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2*mk**6
def argI4s(mk, kj, gm, av, Rp):
        return np.exp(-gm**av*mk**av)*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2*mk**8
def argI0p(mk, kj, gm, lamb2, av, Rp):
        return np.exp(-(gm**av*mk**av + lamb2*mk**2))*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2
def argI1p(mk, kj, gm, lamb2, av, Rp):
        return np.exp(-(gm**av*mk**av + lamb2*mk**2))*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2*mk**2
def argI2p(mk, kj, gm, lamb2, av, Rp):
        return np.exp(-(gm**av*mk**av + lamb2*mk**2))*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2*mk**4
def argI3p(mk, kj, gm, lamb2, av, Rp):
        return np.exp(-(gm**av*mk**av + lamb2*mk**2))*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2*mk**6
def argI4p(mk, kj, gm, lamb2, av, Rp):
        return np.exp(-(gm**av*mk**av + lamb2*mk**2))*np.sinc((Rp*kj*mk)/2)**2*np.sinc(Rp*kj*np.sqrt(1 - mk**2))**2*mk**8
def integrales(sigp, sigt, kv0, kV, av, aV, Rp):
    global I0s
    global I1s
    global I2s
    global I3s
    global I4s
    global I0p
    global I1p
    global I2p
    global I3p
    global I4p
    gm = kj_xi/(kv0*(1 + (kj_xi/kV))**aV)
    lamb2 = (kj_xi**2*(sigp**2 - sigt**2))/2
    I0s=list()
    I1s=list()
    I2s=list()
    I3s=list()
    I4s=list()
    I0p=list()
    I1p=list()
    I2p=list()
    I3p=list()
    I4p=list()
    for i,j,k in zip(kj_xi, gm, lamb2):
        I0saux, err = quad(argI0s, -1, 1, args=(i, j, av, Rp))
        I1saux, err = quad(argI1s, -1, 1, args=(i, j, av, Rp))
        I2saux, err = quad(argI2s, -1, 1, args=(i, j, av, Rp))
        I3saux, err = quad(argI3s, -1, 1, args=(i, j, av, Rp))
        I4saux, err = quad(argI4s, -1, 1, args=(i, j, av, Rp))
        I0paux, err = quad(argI0p, -1, 1, args=(i, j, k, av, Rp))
        I1paux, err = quad(argI1p, -1, 1, args=(i, j, k, av, Rp))
        I2paux, err = quad(argI2p, -1, 1, args=(i, j, k, av, Rp))
        I3paux, err = quad(argI3p, -1, 1, args=(i, j, k, av, Rp))
        I4paux, err = quad(argI4p, -1, 1, args=(i, j, k, av, Rp))
        I0s.append(I0saux)
        I1s.append(I1saux)
        I2s.append(I2saux)
        I3s.append(I3saux)
        I4s.append(I4saux)
        I0p.append(I0paux)
        I1p.append(I1paux)
        I2p.append(I2paux)
        I3p.append(I3paux)
        I4p.append(I4paux)
    I0s=np.array(I0s)
    I1s=np.array(I1s)
    I2s=np.array(I2s)
    I3s=np.array(I3s)
    I4s=np.array(I4s)
    I0p=np.array(I0p)
    I1p=np.array(I1p)
    I2p=np.array(I2p)
    I3p=np.array(I3p)
    I4p=np.array(I4p)
integrales(sigp, sigt, kv0, kV, av, aV, Rp)
def xillist(bF, beta, sigp, sigt, Apeak, kNL, kP, kv0, kV, aNL, aP, av, aV, Rp):
    aj0slist = (1/2)*bF**2*kj_xi**(3/2)*Psmooth(kj_xi)*np.exp((kj_xi/kNL)**aNL - (kj_xi/kP)**aP)*(I0s + 2*beta*I1s + beta**2*I2s)
    aj2slist = (5/4)*bF**2*kj_xi**(3/2)*Psmooth(kj_xi)*np.exp((kj_xi/kNL)**aNL - (kj_xi/kP)**aP)*(-I0s + (3 - 2*beta)*I1s + (6*beta - beta**2)*I2s + 3*beta**2*I3s)
    aj4slist = (9/16)*bF**2*kj_xi**(3/2)*Psmooth(kj_xi)*np.exp((kj_xi/kNL)**aNL - (kj_xi/kP)**aP)*(3*I0s + (6*beta - 30)*I1s + (3*beta**2 - 60*beta + 35)*I2s + (70*beta - 30*beta**2)*I3s + 35*beta**2*I4s)
    aj0plist = (1/2)*Apeak*bF**2*kj_xi**(3/2)*Ppeak(kj_xi)*np.exp((kj_xi/kNL)**aNL - (kj_xi/kP)**aP - ((kj_xi*sigt)**2/2))*(I0p + 2*beta*I1p + beta**2*I2p)
    aj2plist = (5/4)*Apeak*bF**2*kj_xi**(3/2)*Ppeak(kj_xi)*np.exp((kj_xi/kNL)**aNL - (kj_xi/kP)**aP - ((kj_xi*sigt)**2/2))*(-I0p + (3 - 2*beta)*I1p + (6*beta - beta**2)*I2p + 3*beta**2*I3p)
    aj4plist = (9/16)*Apeak*bF**2*kj_xi**(3/2)*Ppeak(kj_xi)*np.exp((kj_xi/kNL)**aNL - (kj_xi/kP)**aP - ((kj_xi*sigt)**2/2))*(3*I0p + (6*beta - 30)*I1p + (3*beta**2 - 60*beta + 35)*I2p + (70*beta - 30*beta**2)*I3p + 35*beta**2*I4p)
    mu0=1/2
    Aj0s=fht(aj0slist, dln_xi, mu0)
    xi0s2list=(1/(2*np.pi**2))*np.sqrt(np.pi/(2*rj_xi**3))*Aj0s
    Aj0p=fht(aj0plist, dln_xi, mu0)
    xi0p2list=(1/(2*np.pi**2))*np.sqrt(np.pi/(2*rj_xi**3))*Aj0p 
    mu2=5/2
    Aj2s=fht(aj2slist, dln_xi, mu2)
    xi2s2list=-(1/(2*np.pi**2))*np.sqrt(np.pi/(2*rj_xi**3))*Aj2s
    Aj2p=fht(aj2plist, dln_xi, mu2)
    xi2p2list=-(1/(2*np.pi**2))*np.sqrt(np.pi/(2*rj_xi**3))*Aj2p
    mu4=9/2
    Aj4s=fht(aj4slist, dln_xi, mu4)
    xi4s2list=(1/(2*np.pi**2))*np.sqrt(np.pi/(2*rj_xi**3))*Aj4s
    Aj4p=fht(aj4plist, dln_xi, mu4)
    xi4p2list=(1/(2*np.pi**2))*np.sqrt(np.pi/(2*rj_xi**3))*Aj4p
    return xi0s2list, xi0p2list, xi2s2list, xi2p2list, xi4s2list, xi4p2list
def Lg2(miuk):
        return (3*miuk**2 - 1)/2
def Lg4(miuk):
        return (35*miuk**4 - 30*miuk**2 + 3)/8
def xicosmo(rj, miuk, alphapl, alphapp):
    Lg0=1.0
    xismooth=Lg0*g0s(rj) + Lg2(miuk)*g2s(rj) + Lg4(miuk)*g4s(rj)
    alpha_ani=np.sqrt(alphapl**2*miuk**2 + alphapp**2*(1-miuk**2))
    r_ani=alpha_ani*rj
    miuk_ani=(alphapl/alpha_ani)*miuk
    xipeak=Lg0*g0p(r_ani) + Lg2(miuk_ani)*g2p(r_ani) + Lg4(miuk_ani)*g4p(r_ani)
    xicosmovalue=xismooth+xipeak
    return xicosmovalue

def xifull(rjlist, miuklist, bF, beta, sigp, sigt, Apeak, kNL, kP, kv0, kV, aNL, aP, av, aV, Rp, alphapl, alphapp):
    xi0s2list, xi0p2list, xi2s2list, xi2p2list, xi4s2list, xi4p2list=xillist(bF, beta, sigp, sigt, Apeak, kNL, kP, kv0, kV, aNL, aP, av, aV, Rp)
    global g0s
    g0s=interp1d(rj_xi, xi0s2list, kind='cubic', bounds_error=False)
    global g0p
    g0p=interp1d(rj_xi, xi0p2list, kind='cubic', bounds_error=False)
    global g2s
    g2s=interp1d(rj_xi, xi2s2list, kind='cubic', bounds_error=False)
    global g2p
    g2p=interp1d(rj_xi, xi2p2list, kind='cubic', bounds_error=False)
    global g4s
    g4s=interp1d(rj_xi, xi4s2list, kind='cubic', bounds_error=False)
    global g4p
    g4p=interp1d(rj_xi, xi4p2list, kind='cubic', bounds_error=False)
    xicoslist = list()
    for rj, miuk in zip(rjlist, miuklist):
        xicos = xicosmo(rj, miuk, alphapl, alphapp)
        xicoslist.append(xicos)
    return xicoslist

