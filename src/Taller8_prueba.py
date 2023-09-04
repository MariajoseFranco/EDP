import numpy as np
import matplotlib.pyplot as plt

def punto2a():
    x = 0
    t = 0
    h = 0.05
    k = 0.0012
    #k = 0.0013
    tabla = []
    aux0 = []
    aux1 = []
    aux1.append(0)
    tabla.append(['t','x','u(x)'])
    while x<1:
        if 0<=x and x<=0.5:
            t0 = lambda x: 2*x
        elif 0.5<x and x<=1:
            t0 = lambda x: 2-2*x
        U00 = t0(x)
        aux0.append(U00)
        tabla.append([t,x,U00])
        U10 = t0(x+h)
        U20 = t0(x+2*h)
        U11 = U10 + (k/(h**2))*(U00-2*U10+U20)
        aux1.append(U11)
        x = x + h
    aux1.remove(aux1[20])
    aux0.append(0)
    aux1.append(0)
    tabla.append([t,x,0])
    t = k
    while t<=50*k:
        tabla.append(['t','x','u(x)'])
        x = 0
        aux2 = []
        aux2.append(0)
        i = 0
        tabla.append([t,x,0])
        x = x + h
        while i<=18:
            U00 = aux1[i]
            U10 = aux1[i+1]
            tabla.append([t,x,U10])
            U20 = aux1[i+2]
            U11 = U10 + (k/(h**2))*(U00-2*U10+U20)
            aux2.append(U11)
            x = x + h
            i = i + 1
        aux2.append(0)
        tabla.append([t,x,0])
        t = t + k
        aux1 = aux2
    for row in tabla:
        print("{:>5} {:>30} {:>30}".format(*row))


def punto3():
    x = 0
    t = 0
    h = 0.05
    k = 0.0012
    #k = 0.0013
    tabla = []
    aux0 = []
    aux1 = []
    exis = []
    aux1.append(0)
    tabla.append(['t','x','u(x)'])
    while x<1:
        exis.append(x)
        if 0<=x and x<=0.5:
            t0 = lambda x: 2*x
        elif 0.5<x and x<=1:
            t0 = lambda x: 2-2*x
        U00 = t0(x)
        aux0.append(U00)
        tabla.append([t,x,U00])
        U10 = t0(x+h)
        U20 = t0(x+2*h)
        U11 = U10 + (k/(h**2))*(U00-2*U10+U20)
        aux1.append(U11)
        x = x + h
    exis.append(x)
    aux1.remove(aux1[20])
    aux0.append(0)
    plt.plot(exis,aux0)
    aux1.append(0)
    tabla.append([t,x,0])
    t = k
    while t<=50*k:
        tabla.append(['t','x','u(x)'])
        x = 0
        aux2 = []
        aux2.append(0)
        i = 0
        tabla.append([t,x,0])
        x = x + h
        while i<=18:
            U00 = aux1[i]
            U10 = aux1[i+1]
            tabla.append([t,x,U10])
            U20 = aux1[i+2]
            U11 = U10 + (k/(h**2))*(U00-2*U10+U20)
            aux2.append(U11)
            x = x + h
            i = i + 1
        aux2.append(0)
        if t==0.0012 or t==0.011999999999999999 or t==0.029999999999999995 or t==0.047999999999999994 or t==0.05999999999999999:
            plt.plot(exis,aux1)
        tabla.append([t,x,0])
        t = t + k
        aux1 = aux2
    for row in tabla:
        print("{:>5} {:>30} {:>30}".format(*row))
    plt.ylabel('u(x)')
    plt.xlabel('x')
    plt.title('con k = 0.0012')
    plt.show()

def punto31():
    x = 0
    t = 0
    h = 0.05
    #k = 0.0012
    k = 0.0013
    tabla = []
    aux0 = []
    aux1 = []
    exis = []
    aux1.append(0)
    tabla.append(['t','x','u(x)'])
    while x<1:
        exis.append(x)
        if 0<=x and x<=0.5:
            t0 = lambda x: 2*x
        elif 0.5<x and x<=1:
            t0 = lambda x: 2-2*x
        U00 = t0(x)
        aux0.append(U00)
        tabla.append([t,x,U00])
        U10 = t0(x+h)
        U20 = t0(x+2*h)
        U11 = U10 + (k/(h**2))*(U00-2*U10+U20)
        aux1.append(U11)
        x = x + h
    exis.append(x)
    aux1.remove(aux1[20])
    aux0.append(0)
    plt.plot(exis,aux0)
    aux1.append(0)
    tabla.append([t,x,0])
    t = k
    while t<=51*k:
        tabla.append(['t','x','u(x)'])
        x = 0
        aux2 = []
        aux2.append(0)
        i = 0
        tabla.append([t,x,0])
        x = x + h
        while i<=18:
            U00 = aux1[i]
            U10 = aux1[i+1]
            tabla.append([t,x,U10])
            U20 = aux1[i+2]
            U11 = U10 + (k/(h**2))*(U00-2*U10+U20)
            aux2.append(U11)
            x = x + h
            i = i + 1
        aux2.append(0)
        if t==0.0013 or t==0.012999999999999998 or t==0.03249999999999999 or t==0.052000000000000025 or t==0.06500000000000004:
            plt.plot(exis,aux1)
        tabla.append([t,x,0])
        t = t + k
        aux1 = aux2
    for row in tabla:
        print("{:>5} {:>30} {:>30}".format(*row))
    plt.ylabel('u(x)')
    plt.xlabel('x')
    plt.title('con k = 0.0013')
    plt.show()

punto31()
#punto2b1()
#punto2b2()