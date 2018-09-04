
# coding: utf-8

# # Optimal velocity model with  a possible extension to a Multiple car-following model
# This program simulates traffic flow using the optimal velocity  model (OVM) proposed by [Bando et al. (1995)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.51.1035). Furthermore, it offers the possibility to extend the driving behaviour to a multiple car-following (MCF) model with a exponential weighting function based on the model of [Peng and Sun, 2010](https://www.sciencedirect.com/science/article/pii/S0375960110001805). 

# ### Compile Fortran module
# The computational expensive calculations can be done using the fortran module when the optimal velocity function (OVF) is set to `tanh`. The following cell compiles the fortran code in the file `f90_subroutines.f90` and builds the extension module ` f90_subroutines`.

# In[53]:


import os
os.system("f2py -c  f90_subroutines.f90 -m f90_subroutines")


# ### Import necessary modules
# The following cell imports necessary modules. The module `functions` contains all functions and routines for the integration of the model. In the module `plot` some pre-designed plots are defined. 

# In[54]:


# import pyhton modules 
import functions as fc
import plot as plot
from pathes import outpath

# import other modules
import numpy as np
import matplotlib.pyplot as plt

import time
import gc


# ### Set parameters for the model
# `N` - number of cars
# 
# `L`    - length of the circuit
# 
# `a` -  sensitivity
# 
# `h`    - parameter in the OV function (inflection point)
# 
# `tmax` - integration time
# 
# `dt`   - time step for the numerical integration
# 
# `ovf`  -  OV function
# 
# `m`    - box size for mesoscale variables 
# 
# `v0` - velocity scale
# 
# `box`  - box in front, middle or back of a car
# 
# `weight` - weighting function
# 
# `model` - OVM, OVM_rho, OVM_Delta_x_relax2J, OVM_rho_relax2J, relax2J
# 
# `lambda`      - relaxing strength

# In[55]:


parameters = {
        "N":100,
        "L":200,
        "a":1.,
        "h":2.,
        "tmax":1000, 
        "dt" : 0.1,
        "v0":1.,
        "ovf":"tanh",
        "m": 1.,
        "box":"front",
        "weight":"exp",
        "weight_parameter":0.5,
        "model":"OVM_rho_relax2J",
        "lambda": 0.0,
        "noise":0.0
        }

parameters["xpert"] = np.zeros(parameters["N"])
parameters["xpert"][0] = 0.1


# ### Model simulation
# 
# 

# In[56]:


t_start = time.time() 

ovm = fc.ovm(parameters)
ovm.initCars(option="normal")
ovm.integrate(kernel="python") # note that if kernel is fortran model automatically is "OVM_rho_relax2J" and ovf is "tanh"

t_end = time.time()

print "Integration time:" + str(t_end-t_start)


# In[ ]:


plt.close("all")

plot.ovm_small_panal(ovm)

gc.collect()

