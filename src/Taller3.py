#Taller 3
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
import sympy as sp
import math
from numpy import *

def punto5c():
    for i in [-6,0,6]:
        x0 = np.arange(-3+i,0+i,0.01)
        y0 = [0 for j in range(300)]
        x1 = np.arange(0+i,3+i,0.01)
        y1 = [1 for j in range(300)]
        plt.plot(x0,y0,color="blue")
        plt.plot(x1,y1,color="blue")
    x = np.arange(-9,9,0.01)
    y = 1/2
    for n in range(1,100):
        y = y + (2/np.pi)*(1/(2*n-1))*np.sin(((2*n-1)*np.pi*x)/3)
    plt.plot(x,y,color="red",label="Fourier")
    sumParciales = [2,5,10,20,40]
    for n in sumParciales:
        y = 1/2
        for i in range(1,n+1):
            y = y + (2/np.pi)*(1/(2*i-1))*np.sin(((2*i-1)*np.pi*x)/3)
        plt.plot(x,y,label="n=%s" % n)
    plt.legend()
    plt.show()

def punto5d():
    x = sp.Symbol('x')

    sumParciales = [2,5,10,20,40]
    for i in sumParciales:
        y = 1/2
        for n in range(1,i+1):
            y = y + (2/np.pi)*(1/(2*n-1))*sp.sin((2*n-1)*np.pi*x/3)

        fParte1 = lambda x: pow(0-eval(str(y)),2)
        fParte2 = lambda x: pow(1-eval(str(y)),2)

        errorParte1 = quad(fParte1,-3,0)
        errorParte2 = quad(fParte2,0,3)

        e = math.sqrt(errorParte1[0]+errorParte2[0])
        print('Error de la suma parcial n=%s:' % i,e)

#punto5c()
#punto5d()