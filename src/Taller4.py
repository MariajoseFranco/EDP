#Taller 4
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

def punto1():
    sumParciales = [2,4,16,64]
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,4,10]
    cont_subplot = 1
    for t in tiempo:
        plt.subplot(2,2,cont_subplot)
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        for n in sumParciales:
            y = 0
            for i in range(1,n+1):
                y = y + ((-4/pow(np.pi*i,5))*pow(-1,i)*(pow(i*np.pi,2)*t-1+np.exp(-pow(i*np.pi,2)*t))-(32/pow(np.pi*i,3))*np.exp(-pow(np.pi*i,2)*t)*(2*pow(-1,i)+1))*np.sin(np.pi*i*x/2)
            plt.title("t = %s" % t)
            #Graficas sumas parciales
            plt.plot(x,y,label="n=%s" % n)
            plt.legend()
        cont_subplot += 1
    plt.tight_layout()
    plt.show()

def punto2():
    sumParciales = [2,4,16,64]
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,4,10]
    cont_subplot = 1
    for t in tiempo:
        plt.subplot(2,2,cont_subplot)
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        for n in sumParciales:
            y = (1/2)*((-np.pi*np.cos(t))+np.pi+(4/np.pi))
            for i in range(2,n+1):
                y = y + np.cos(i*x)*((2/(np.pi*pow(i,2)*(pow(i,4)+1))*(pow(-1,i)-1)*((pow(i,2)*np.sin(t))-np.cos(t)+
                                  np.exp(-t*pow(i,2))))+((2/(np.pi*(1-pow(i,2))))*np.exp(-t*pow(i,2))*(pow(-1,i)+1)))
            plt.title("t = %s" % t)
            #Graficas sumas parciales
            plt.plot(x,y,label="n=%s" % n)
            plt.legend()
        cont_subplot += 1
    plt.tight_layout()
    plt.show()

def punto3():
    sumParciales = [2,4,16,64]
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,4,10]
    cont_subplot = 1
    for t in tiempo:
        plt.subplot(2,2,cont_subplot)
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        for n in sumParciales:
            y = 0
            for i in range(1,n+1):
                f = lambda X: (2/np.pi)*np.exp(2*X)*X*(np.pi-X)*np.sin(i*X)
                integ = quad(f,0,np.pi)
                integral = integ[0]
                y = y + np.exp(-2*(x+t))*integral*np.sin(i*x)*np.exp(-t*pow(i,2))
            plt.title("t = %s" % t)
            #Graficas sumas parciales
            plt.plot(x,y,label="n=%s" % n)
            plt.legend()
        cont_subplot += 1
    plt.tight_layout()
    plt.show()

def punto4():
    sumParciales = [2,4,16,64]
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,4,10]
    cont_subplot = 1
    for t in tiempo:
        plt.subplot(2,2,cont_subplot)
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        for n in sumParciales:
            y = 0
            for i in range(1,n+1):
                y = y + np.sin((i*np.pi*x)/3)*(-54/(pow(i*np.pi,4)))*(pow(-1,i)-1)*np.sin((2*i*np.pi*t)/3)
            plt.title("t = %s" % t)
            #Graficas sumas parciales
            plt.plot(x,y,label="n=%s" % n)
            plt.legend()
        cont_subplot += 1
    plt.tight_layout()
    plt.show()

#punto1()
#punto2()
punto3()
#punto4()