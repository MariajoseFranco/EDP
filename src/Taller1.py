#Taller 1
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from numpy import *
import sympy as sp

def punto2bFourier():
    #f(x)
    x0 = np.arange(-np.pi,0,0.01)
    y0 = (2*x0/np.pi)+2
    x1 = np.arange(0,np.pi,0.01)
    y1 = [2 for i in range(315)]

    #Sumas Parciales y Serie de Fourier
    sumParciales = [2,4,8,16,32,64]
    x = np.arange(-np.pi,np.pi,0.01)
    count = 1
    for n in sumParciales:
        y = 3/2
        for i in range(1,n+1):
            y = y + 2*np.cos(i*x)*(1-pow(-1,i))/pow(np.pi*i,2) - 2*np.sin(i*x)*pow(-1,i)/(np.pi*i)
        #Confrontacion de la grafica f(x) con cada una de las graficas de la suma parcial
        plt.figure(count)
        plt.title("Serie de Fourier")
        #Grafica f(x)
        plt.plot(x0,y0,color="red")
        plt.plot(x1,y1,color="red",label="f(x)")
        #Graficas sumas parciales
        plt.plot(x,y,label="n=%s" % n)
        plt.legend()
        count = count + 1
    plt.show()

def punto2cErrorFourier():
    x = sp.Symbol('x')
    print('Errores de las Sumas Parciales')

    #Sumas Parciales
    sumParciales = [2,4,8,16,32,64]
    for n in sumParciales:
        y = 3/2
        for i in range(1,n+1):
            y = y + 2*sp.cos(i*x)*(1-pow(-1,i))/pow(sp.pi*i,2) - 2*sp.sin(i*x)*pow(-1,i)/(sp.pi*i)

        #Errores sumas parciales por partes: fParte1 = tramo de (-pi,0), fParte2 = tramo de (0,pi)
        fParte1 = lambda x: pow(((1/np.pi)*2*x+2)-eval(str(y)),2)
        fParte2 = lambda x: pow(2-eval(str(y)),2)

        errorParte1 = quad(fParte1,-np.pi,0)
        errorParte2 = quad(fParte2,0,np.pi)

        e = sp.sqrt(errorParte1[0]+errorParte2[0])
        print('Error de la suma parcial n=%s:' % n,e)

punto2bFourier()
#punto2cErrorFourier()