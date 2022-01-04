#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 19:48:10 2022

@author: mbajec
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

L = 420
A = 0.1
B = 20
C = 0.6
data = np.array([A*(i // 12) + B + C*np.sin(np.pi*(i % 12)/6) + np.random.normal() for i in range(L)])

# =============================================================================
# 
# =============================================================================


leta = np.arange(0,35,1)
meseci = np.arange(0,12,1)

leta_M = np.array([leta[i // 12] for i in range(420)])
meseci_M = np.sin(np.pi*np.array([meseci[i%12] for i in range(420)])/6)

M = np.array([leta_M,
             np.ones(420),
             meseci_M])

# =============================================================================
# 
# =============================================================================

# fitParams = np.array([a, b, c])

result = LA.lstsq(M.T, data)
fitParams = result[0]


# =============================================================================
# 
# =============================================================================
#%%
N=420

y = fitParams[0]*leta_M + fitParams[1]*np.ones(420) + fitParams[2]*np.sin(meseci_M*np.pi/6)
residuals = np.array([data[i] - y[i] for i in range(420)])


fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(data[:N], 'ko', markersize=2)
ax.plot(y[:N], 'r--')

ax2 = fig.add_subplot(212)
ax2.plot(residuals)

figH = plt.figure()
axH = figH.add_subplot(111)
axH.hist(residuals, bins=40)


