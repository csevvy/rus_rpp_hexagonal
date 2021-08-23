# -*- coding: utf-8 -*-
"""
TiB2 RUS Code Execution

@author: Chris
"""
import time
import Rus_Module as r
#This is the material properties used for the Hexagonal material considered.
#Values below are for TiB2 based on ledbetter et. al.
c11=654e9 #Pascals
c33=458e9
c44=262e9
c13=95e9
c12=49e9
rho=4520 #kg/m3

# This is the dimensions of the rectangular prism sample and the nn decides 
# the dimension of the functional basis (higher nn means better approximation)
nn=10
d1=5e-3 #These are the dimensions of a RPP sample (in meters)
d2=3e-3
d3=2e-3
t = time.time()
(freqs,eigVec)=r.rus_function_hex(c11, c33, c44, c13, c12, d1, d2, d3, rho, nn)
print(time.time()-t)
t = time.time()
r.plot_modeshape(1, d1, d2, d3, nn, eigVec)
r.plot_modeshape(20, d1, d2, d3, nn, eigVec)
r.plot_modeshape(50, d1, d2, d3, nn, eigVec)
print(time.time()-t)