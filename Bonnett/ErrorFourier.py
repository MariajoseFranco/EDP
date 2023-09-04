import sympy as sp
import numpy as np
from scipy.integrate import quad
from numpy import *
import math as math


a = [2,5,10,20,40] #Define all n values
x = sp.Symbol('x')

#Build the function to be integrated
errors = []

for i in a:
    y = 1/2
    for j in range(1,i+1):
        y = y + 2*np.pi*(1/(2*j-1))*np.sin((2*j-1)*np.pi*x)/3


    f1 = lambda x:(0 - eval(str(y)))**2
    f2 = lambda x: (x - eval(str(y)))**2
    
    error1 = quad(f1,-np.pi,0)[0]
    error2 = quad(f2,0,np.pi)[0]

    errors.append(math.sqrt(error1+error2))
    
print(errors)
    

# >>> from numpy import *
# >>> a = str(-2.46740110027234*pi - 8.0/pi + 12.5663706143592/pi**2 + 0.392699081698724*pi**2 + 6.4084347431127)
# >>> eval(a)
# 1.2594106133400604