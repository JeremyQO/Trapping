#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:31:37 2022

@author: jeremy
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import scipy.constants as sc


# data = np.load('trap.npz', allow_pickle = True)
# t = data['t']
# pots = data['pots']*1e-3*sc.k
# depths = data['depths']
# poss = data['poss']

# if all(p==0 for p in poss) or all(d==0 for d in depths):
#     print('No trap minima exists')
#     ret = -1e30


# pmax = [argrelextrema(pot, np.greater)[0] for pot in pots]
# tmax = [t[p] for p in pmax]
# pmin = [argrelextrema(pot, np.less)[0] for pot in pots]
# tmin = [t[p] for p in pmin]


# int_a = [p[0]+2 for p in pmax]
# int_b = [int(-(len(t)-p[0])//1.3) for p in pmin]

# pots_r = [pot[int_a[i]: int_b[i]] for i,pot in enumerate(pots)]
# t_r = [t[int_a[i]:int_b[i]] - tmin[i] for i in range(len(int_a))]


# order = 6
# res =  np.array([np.polyfit(t_r[i], pots_r[i], deg=order) for i in range(len(t_r))])
# p = [np.poly1d(r) for r in res]

# # p2 = np.poly1d(res[-2:])
# plt.clf()
# plt.plot(t_r[2], pots_r[2],'.', label="Simulation")
# plt.plot(t_r[2], p[2](t_r[2]), label="Polynomial fit, order %i"%(order))
# plt.plot(t_r[2], res[2][-1]+res[2][-3]*(t_r[2])**2+res[2][-2]*(t_r[2]),label="Harmonic part")
# plt.legend()

# # plt.clf()
# # start = pmax[2][0]
# # plt.plot(t[start:], pots[2][start:],'.')
# # plt.plot(t[start:], p[2](t[start:]-tmin[2]))
# # plt.plot(t[start:], res[2][-1]+res[2][-3]*(t[start:]-tmin[2])**2+res[2][-2]*(t[start:]-tmin[2]))
# ret = np.real(res[:,-3])
# m_Rb = 86.909184*sc.atomic_mass
# freq = np.sqrt(ret*2/m_Rb)/2/sc.pi
# print(freq)

plt.clf()
pot_min = -0.9811002732280614 *1e-3*sc.k
beta = 1.5536 * 2*sc.pi/850e-9
z = np.linspace(-850e-9/20, 850e-9/20, 300)

pot =  np.cos(beta*z)**2*pot_min 
plt.plot(z, pot)








