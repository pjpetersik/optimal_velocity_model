#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:43:14 2018

OVM_functions.py contains all functions for the OVM model

@author: paul
"""

from f90_subroutines import subs as ovmf90
import numpy as np




class ovm(object):
    def __init__(self,parameters):
        """
        parameters of the model simulation are setup
        setup 
        """
        self.update(parameters)    
    
    def update(self,parameters):
            
        self.N =  parameters["N"]      # number of cars
        self.L =  parameters["L"]      # length of circuit
        self.distance = np.arange(0,self.L,1) #array for distance
        
        self.a =  parameters["a"]      # sensitiviy
        self.h =  parameters["h"]      # 
        self.v0 = parameters["v0"] # maximum velocity
        
        self.m    = parameters["m"]  # number of cars in the interaction box (must be even number, if not is n=n-1)
        self.box  = parameters["box"] # box front, middle or back
        self.wf_name = parameters["weight_function"] 
        self.wp = parameters["weight_parameter"] 
        
        self.tmax  = parameters["tmax"] # maximum time
        self.dt    = parameters["dt"]     # time step
        self.iters = abs(int(self.tmax/self.dt))
        self.time  = np.arange(0,self.tmax,self.dt)
        
        self.xpert = parameters["xpert"] # position perturbation
        
        self.ovf = parameters["ovf"] # key for the choice of the OV-function
        self.acceleration_type = parameters["model"]
        self.lambda_v = parameters["lambda"]
        
        # allocate functions
        self.allocate_functions()

# =============================================================================
# Routines        
# =============================================================================
    
    def allocate_functions(self):
        """
        allocate some functions 
        """
        
        # optimal velocity function
        if self.ovf=="tanh":
            self.V=self.ovf_tanh
        if self.ovf=="hs":
            self.V=self.ovf_hs
        if self.ovf=="alg":
            self.V=self.ovf_alg
        
        # box of the considered platoon
        if self.box =="front":
            self.density = self.density_front
            self.flow_velocity = self.flow_velocity_front
        if self.box =="middle":
            self.density = self.density_middle
            self.flow_velocity = self.flow_velocity_middle
        if self.box =="back":
            self.density = self.density_back
            self.flow_velocity = self.flow_velocity_back
        
        # model type
        if self.acceleration_type == "OVM":
            self.acceleration = self.acceleration_OVM
        if self.acceleration_type == "MCF":
            self.acceleration = self.acceleration_MCF

        # weigthing function
        if self.wf_name == "exp":
            self.wf = self.exp_wf
        if self.wf_name == "lin":
            self.wf = self.lin_wf
        
    
    def initCars(self,**kwargs):
        """
        initialise 0th time step
        """  
        
        self.b,self.c,self.f = self.steadyStateFlow(self.L,self.N)  # free flow variable
        
        self.x       = np.zeros(shape=(self.N,self.iters)) # position
        self.dot_x   = np.zeros(shape=(self.N,self.iters)) # velocity
        self.ddot_x  = np.zeros(shape=(self.N,self.iters)) # acceleration
        self.Delta_x = np.zeros(shape=(self.N,self.iters)) # headway
        self.local_rho = np.zeros(shape=(self.N,self.iters)) # local density
        self.local_flow = np.zeros(shape=(self.N,self.iters)) # local density
        self.local_q = np.zeros(shape=(self.N,self.iters)) # local flux-density
        
        
        self.x[:,0]      = np.arange(0,self.L,self.b)[:self.N] # make sure that array is not (accidentlly) to big
        self.dot_x[:,0]  = self.c 
        
        self.ddot_x[:,0] = 0.
        
        self.x[:,0] = self.x[:,0] + self.xpert
        self.Delta_x[:,0]   = self.headway(self.x[:,0],self.L)
        self.local_rho[:,0] = self.density(self.x[:,0],self.Delta_x[:,0])
        self.local_flow[:,0] = self.flow_velocity(self.dot_x[:,0])
    
        
            
    def integrate(self,**kwargs):
        """
        Integrate the model using a fortran or a python kernel 
        """
        
        if kwargs["kernel"]=="fortran":
            for i in range(0,self.iters-1):
                self.integration_procedure_f90(i)
                 
            
        elif kwargs["kernel"]=="python":
            for i in range(0,self.iters-1):
                self.integration_procedure(i)
                
    def integration_procedure(self,i):
        """
        Runge-Kutta 4 integration scheme
        """
        h = self.dt
        k1 = self.acceleration(self.Delta_x[:,i],self.local_flow[:,i],self.local_rho[:,i],self.dot_x[:,i])
        self.dot_x[:,i+1] = self.dot_x[:,i] + k1*h/2
        
        k2 = self.acceleration(self.Delta_x[:,i],self.local_flow[:,i],self.local_rho[:,i],self.dot_x[:,i+1])
        
        self.dot_x[:,i+1] = self.dot_x[:,i] + k2*h/2
        k3 = self.acceleration(self.Delta_x[:,i],self.local_flow[:,i],self.local_rho[:,i],self.dot_x[:,i+1])
        
        self.dot_x[:,i+1] = self.dot_x[:,i] + k3*h
        k4 = self.acceleration(self.Delta_x[:,i],self.local_flow[:,i],self.local_rho[:,i],self.dot_x[:,i+1])
        
        self.ddot_x[:,i+1] = k1 
        
        self.dot_x[:,i+1] = self.dot_x[:,i] + h/6. * (k1 + 2*k2 + 2*k3 + k4) 
        
        self.x[:,i+1]      = self.x[:,i] + self.dot_x[:,i+1] * h
        
        self.x[:,i+1]      = self.x[:,i+1]%self.L

        # Diagnostics
        self.Delta_x[:,i+1]   = self.headway(self.x[:,i+1],self.L)
        self.local_rho[:,i+1] = self.density(self.x[:,i+1],self.Delta_x[:,i+1])
        self.local_flow[:,i+1] = self.flow_velocity(self.dot_x[:,i+1])
        self.local_q[:,i+1] = self.local_rho[:,i+1] * self.local_flow[:,i+1] 

    
    def integration_procedure_f90(self,i):
        """
        using a rk4 scheme from a fortran code
        """
        self.dot_x[:,i+1],self.x[:,i+1],self.Delta_x[:,i+1],self.local_rho[:,i+1],self.local_flow[:,i+1] = \
                ovmf90.rk4(self.a,self.lambda_v,self.v0,self.h,self.L,self.N,self.m,self.wp, \
                           self.local_rho[:,i],self.local_flow[:,i],self.x[:,i],self.dot_x[:,i],self.dt)

# =============================================================================
# Functions        
# =============================================================================    
    def acceleration_MCF(self,Delta_x,local_flow,local_rho,dot_x):
        """
        returns the accelaration of a car as relaxation to local flow
        """
        # using pyhton
        ddotx = self.a*(self.V(1/local_rho) - dot_x) + self.lambda_v * (local_flow - dot_x)
        return ddotx
    
    def acceleration_OVM(self,Delta_x,local_flow,local_rho,dot_x):
        """
        returns the accelaration of a car as function of Delta x
        """
        return self.a*(self.V(Delta_x) - dot_x)
    
    
    def headway(self,x,L):
        Dx = np.zeros(self.N)
        Dx[:-1] = ((x[1:] - x[:-1])+L)%L
        Dx[-1] = ((x[0] - x[-1])+L)%L
        return Dx #(np.roll(x,-1)-x+L)%L
    
    def ovf_tanh(self,Delta_x):
        """
        OV - function as in Bando et al (1995)
        Legal velocity - V(Delta_x)
        Delta_x - headway to the car in front
        """
        
        return self.v0*(np.tanh(Delta_x - self.h) + np.tanh(self.h))
    
    def ovf_hs(self,Delta_x):
        """
        OV - function as in Sugiyama and Yamada (1997)
        Legal velocity - V(Delta_x)
        Delta_x - headway to the car in front
        """
        return self.v0*(np.heaviside(Delta_x - self.h,1))
    
    def ovf_alg(self,Delta_x):
        """
        OV - function as in Orosz (2005)
        Legal velocity - V(Delta_x)
        Delta_x - headway to the car in front
        """
        ovf_return = np.zeros(self.N)
        ovf_return[:] = self.v0*np.divide(np.power(Delta_x - 1,3),1+np.power(Delta_x - 1,3))
        
        index = np.where(Delta_x<=1)
        ovf_return[index] = 0
        
        return ovf_return
       
    def steadyStateFlow(self,L,N):
        """
        Returns parameters b, c and f of  a steady state flow.
        Input: 
            L - length of circuit
            N - number of cars
        Returns:
            b - constant spacing
            c - constant velocity
            f - derivative V(b) 
        """
        b = float(L)/float(N)
        c = self.V(b)
        f = 1 - np.tanh(b)**2
        return b,c,f
    
    def density_front(self,x,Dx):
        """
        compute the local density for each car
        x is unused but can't be excluded because self.density(x,Dx) needs to have x and Dx
        as argument for density back and middle 
        
        the function returns the weighted avergage of Dx times the number of cars
        
        old box size: box_size_old = (np.roll(x,-car_number) -x)%self.L
        """
        car_number = int(self.m)
        # extend array for periodic boundary  
        Dx_extended = np.append(Dx,Dx[:car_number-1])
        
        # prepare weights so that weighting can be done by a convolution
        v_values = self.wf(np.arange(car_number))
        
        # weightes  
        v = v_values/sum(v_values)*car_number
        
        # apply convolution of headways with weights
        box_size = (np.convolve(Dx_extended, v, mode='valid'))#%self.L with the modulo this leads to results as if box_size_old is used 
        
        # compute local front rho
        rho = float(car_number)/box_size
        return rho
    
    def density_middle(self,x,Dx):
        """
        compute the local density for each car
        """
        car_number = int(self.m/2)
        # density middle
        
        box_size = (np.roll(x,-car_number) -x)%self.L + (x -np.roll(x,car_number))%self.L
        rho = 2*float(car_number)/box_size
        return rho
    
    def density_back(self,x,Dx):
        """
        compute the local density for each car
        """
        car_number = int(self.m)
        #density back
        box_size = (x -np.roll(x,car_number))%self.L 
        rho = float(car_number)/box_size
        return rho
        
    def flow_velocity_front(self,dotx):
        """
        compute the local flow velocity for each car using moving averages with 
        periodic boundary conditions
        """
        # flow velocity front
        car_number = int(self.m)
        dotx_extended = np.append(dotx,dotx[:car_number-1])
        
        a = dotx_extended
        
        v_function = self.wf(np.arange(car_number))
       
        v = v_function/sum(v_function)
        
        dotx_flow = np.convolve(a, v, mode='valid')
        dotx_flow = np.roll(dotx_flow,-1)
        return dotx_flow
    
    def flow_velocity_middle(self,dotx):
        """
        compute the local flow velocity for each car using moving averages with 
        periodic boundary conditions
        """
        # flow velocity middle
        car_number = int(self.m/2)
        dotx_extended = np.append(dotx[-car_number:],dotx)
        dotx_extended = np.append(dotx_extended,dotx[:car_number])         
        dotx_flow = np.convolve(dotx_extended, np.ones((2*car_number+1,))/(2*car_number+1), mode='valid')
        return dotx_flow

    def flow_velocity_back(self,dotx):
        """
        compute the local flow velocity for each car using moving averages with 
        periodic boundary conditions
        """
        #flow velocity back
        car_number = int(self.m)
        dotx_extended = np.append(dotx[-car_number:],dotx)
        dotx_flow = np.convolve(dotx_extended, np.ones((car_number+1,))/(car_number+1), mode='valid')
        return dotx_flow
    
    def exp_wf(self,argument):
        return np.exp(self.wp*argument)     
    
    def lin_wf(self,argument):
        return self.wp*argument + 1    