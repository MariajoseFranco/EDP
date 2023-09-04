#Taller 5
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

def punto1():
    sumParciales = [2,4,8,16,32]
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,2,3,4,10,20]
    cont_subplot = 1
    for t in tiempo:
        plt.subplot(2,4,cont_subplot)
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        for n in sumParciales:
            y = x+pow(t,2)+12*np.exp(-t/4)*np.sin(x/2)+(4*t+12-12*np.exp(-9*t/4)+4*np.exp(-9*t/4))*np.sin(3*x/2)
            for i in range(1,n+1):
                y = y + (12/(2*i+1))*np.exp(-pow((2*i+1)/2,2)*t)*np.sin((2*i+1)*x/2)
            plt.title("t = %s" % t)
            #Graficas sumas parciales
            plt.plot(x,y,label="n=%s" % n)
            plt.legend()
        cont_subplot += 1
    plt.tight_layout()
    plt.show()

def punto2():
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,2,3,4,10,20]
    cont_subplot = 1
    for t in tiempo:
        u=x*np.sin(t)+t+1+np.exp(-4*t*np.pi**2)*np.cos(2*np.pi*x)
        plt.subplot(2,4,cont_subplot)
        plt.title('u(x,t) en t='+ str(t))
        plt.xlabel('x'); plt.ylabel('u(x,'+str(t)+')')
        plt.plot(x,u)
        cont_subplot=cont_subplot+1
    plt.tight_layout()
    plt.show()

def punto3():
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,2,3,4,10,20]
    cont_subplot = 1
    for t in tiempo:
        u=(2*(t**2)/np.pi)-np.cos(t)+1+(np.sin(6*t)/6)*np.cos(3*x)+x*np.cos(t)-x+(x**2)/(2*np.pi)
        plt.subplot(2,4,cont_subplot)
        plt.title('u(x,t) en t='+ str(t))
        plt.xlabel('x'); plt.ylabel('u(x,'+str(t)+')')
        plt.plot(x,u)
        cont_subplot=cont_subplot+1
    plt.tight_layout()
    plt.show()

def punto4():
    sumParciales = [2,4,8,16,32]
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,2,3,4,10,20]
    cont_subplot = 1
    for t in tiempo:
        plt.subplot(2,4,cont_subplot)
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        for n in sumParciales:
            y = x**2/2+13*t+1/2
            for i in range(1,n+1):
                y = y + (-4/(np.pi**2))*(np.exp(-13*(2*i-1)**2*np.pi**2*t)*np.cos((2*n-1)*np.pi*x))/((2*i-1)**2)
            plt.title("t = %s" % t)
            #Graficas sumas parciales
            plt.plot(x,y,label="n=%s" % n)
            plt.legend()
        cont_subplot += 1
    plt.tight_layout()
    plt.show()

def punto5():
    x = np.arange(-3,3,0.01)
    tiempo = [0,1,2,3,4,10,20]
    cont_subplot = 1
    for t in tiempo:
        u=t+1/2-(1/5)*np.cos(3*t)*np.cos(3*x)+(1/5)*np.cos(2*t)*np.cos(3*x)+(1/2)*np.cos(2*t)*np.cos(2*x)
        plt.subplot(2,4,cont_subplot)
        plt.title('u(x,t) en t='+ str(t))
        plt.xlabel('x'); plt.ylabel('u(x,'+str(t)+')')
        plt.plot(x,u)
        cont_subplot=cont_subplot+1
    plt.tight_layout()
    plt.show()

#punto1()
#punto2()
#punto3()
#punto4()
punto5()