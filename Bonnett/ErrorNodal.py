import sympy as sp
import numpy as np
from scipy.integrate import quad
from numpy import *
import math as math


a = [2,5,8,12,25,50] #Define all n values
x = sp.Symbol('x')

#Build the function to be integrated
errors = []

for i in a:
    y = sp.pi/4
    for j in range(1,i+1):
        y = y + (((-1)**j - 1)/(np.pi * j**2))*np.cos(j*(1/2)) + (((-1)**(j+1))/(j))*np.sin(j*(1/2))

    print(abs(0.5 - eval(str(y))))