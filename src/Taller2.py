#Taller 2
import matplotlib.pyplot as plt
import numpy as np

def punto1sen():
    #Graficar la extension periodica impar de la serie de fourier de senos
    for i in [-2*np.pi,0,2*np.pi]:
        x0 = np.arange(0+i,(np.pi/2)+i,0.01)
        y0 = [1 for j in range(158)]
        x1 = np.arange((np.pi/2)+i,np.pi+i,0.01)
        y1 = [2 for j in range(158)]
        #ext
        x2 = np.arange(0+i,(-np.pi/2)+i,-0.01)
        y2 = [-1 for j in range(158)]
        x3 = np.arange((-np.pi/2)+i,-np.pi+i,-0.01)
        y3 = [-2 for j in range(158)]
        plt.title("Serie de Fourier de Senos")
        plt.plot(x0,y0,color="blue")
        plt.plot(x1,y1,color="blue")
        plt.plot(x2,y2,color="blue")
        plt.plot(x3,y3,color="blue")
    x = np.arange(-3*np.pi,3*np.pi,0.01)
    y = 0
    for n in range(1,20):
        y = y + (2/(np.pi*n))*(1-2*pow(-1,n)+np.cos(np.pi*n/2))*np.sin(n*x)
    plt.plot(x,y,color="red",label="fourier")
    plt.legend()
    plt.show()

def punto1cos():
    #Graficar la extension periodica par de la serie de fourier de cosenos
    for i in [-2*np.pi,0,2*np.pi]:
        x0 = np.arange(0+i,(np.pi/2)+i,0.01)
        y0 = [1 for j in range(158)]
        x1 = np.arange((np.pi/2)+i,np.pi+i,0.01)
        y1 = [2 for j in range(158)]
        #ext
        x2 = np.arange(0+i,(-np.pi/2)+i,-0.01)
        y2 = [1 for j in range(158)]
        x3 = np.arange((-np.pi/2)+i,-np.pi+i,-0.01)
        y3 = [2 for j in range(158)]
        plt.title("Serie de Fourier de Cosenos")
        plt.plot(x0,y0,color="blue")
        plt.plot(x1,y1,color="blue")
        plt.plot(x2,y2,color="blue")
        plt.plot(x3,y3,color="blue")
    x = np.arange(-3*np.pi,3*np.pi,0.01)
    y = 3/2
    for n in range(1,25):
        y = y - (2/np.pi)*np.sin(np.pi*n/2)*(np.cos(n*x))/n
    plt.plot(x,y,color="red",label="fourier")
    plt.legend()
    plt.show()

def punto2():
    #Graficar la extension periodica impar de la serie de fourier de senos
    x0 = np.arange(0,np.pi,0.01)
    y0 = x0*(np.pi-x0)
    x1 = np.arange(0,-np.pi,-0.01)
    y1 = x1*(np.pi+x1)
    plt.plot(x0,y0,color="blue")
    plt.plot(x1,y1,color="blue")
    for i in [-2*np.pi,0,2*np.pi]:
        x0 = np.arange(0+i,np.pi+i,0.01)
        #ext
        x1 = np.arange(0+i,-np.pi+i,-0.01)
        plt.title("Serie de Fourier de Senos")
        plt.plot(x0,y0,color="blue")
        plt.plot(x1,y1,color="blue")
    x = np.arange(-3*np.pi,3*np.pi,0.01)
    y = 0
    for n in range(1,100):
        y = y + (8/np.pi)*np.sin((2*n-1)*x)/pow(2*n-1,3)
    plt.plot(x,y,color="red",label="fourier")
    plt.legend()
    plt.show()

#punto1sen()
#punto1cos()
#punto2()