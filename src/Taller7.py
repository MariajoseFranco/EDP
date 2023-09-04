#Taller 7
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy.integrate import quad
import math
import sympy as sym


def punto2_tablas_aproximacion_y_graficas():
    f= lambda x, y:-x*y-np.cos(6*x)*exp(-1/2*x**2)
    X=np.linspace(0,1,100)
    solucion=-1/6*exp(-1/2*X**2)*(np.sin(6*X)-6)
    for j in list(range(1,11)):
        cont=0
        todas_x=[]
        todas_y=[]
        tabla=[]
        x0=0
        y0=1
        yprima=f(x0,y0)
        h=2**(-j)
        todas_x.append(x0)
        todas_y.append(y0)
        tabla.append(['i','h','   xi','  Yi', '     Yi\''])
        tabla.append([cont,h,x0,y0,yprima])
        while x0<1:
            x1=x0+h
            y1=y0+h*f(x0,y0)
            todas_x.append(x1)
            todas_y.append(y1)
            yprima=f(x1,y1)
            cont=cont+1
            tabla.append([cont,h,x1,y1,yprima])
            x0=x1
            y0=y1
        for row in tabla:
            print(" {:>20} {:>5} {: >20} {: >20} {: >20}".format(*row))
        plt.plot(todas_x,todas_y,label="exacta")
        plt.plot(X,solucion,label="aproximada")
        plt.legend(loc="upper right")
        plt.xlabel('x'); plt.ylabel("y");plt.title("Solución real vs aproximada con h="+str(h))
        plt.show()

def punto2_error_y_graficas_error():
    f= lambda x, y:-x*y-np.cos(6*x)*exp(-1/2*x**2)
    solucion=lambda x:-1/6*exp(-1/2*x**2)*(np.sin(6*x)-6)
    tabla=[]
    tabla.append(['j','h','Eh','a'])
    cont=0
    todas_las_h=[]
    todos_los_Eh=[]
    for j in list(range(1,11)):
        cont=cont+1
        x0=0
        y0=1
        h=2**(-j)
        todas_las_h.append(h)
        suma_errores=0
        while x0<1:
            x1=x0+h
            y1=y0+h*f(x0,y0)
            m=(y1-y0)/(x1-x0)
            v=lambda x:m*(x-x0)+y0
            integrando=lambda x:abs(solucion(x)-v(x))**2
            integral1,error_aprox=quad(integrando,x0,x1)
            suma_errores=suma_errores+integral1
            x0=x1
            y0=y1
        Eh=math.sqrt(suma_errores)
        todos_los_Eh.append(Eh)
        if cont>1:
            filaAnt=tabla[cont-1]
            h_ant=filaAnt[1]
            Eh_ant=filaAnt[2]
            a=math.log(Eh/Eh_ant)/math.log(h/h_ant)
        else:
            a='n/a'
        tabla.append([cont,h,Eh,a])
    for row in tabla:
        print("{:>17} {:>17} {:>17} {:>17} ".format(*row))
    plt.plot(todas_las_h,todos_los_Eh)
    plt.title('Eh vs h')
    plt.xlabel=("h");plt.ylabel=("Eh")
    plt.show()
    plt.loglog(todas_las_h,todos_los_Eh)
    plt.title('Log(Eh) vs Log(h)')
    plt.xlabel=("Log(h)");plt.ylabel=("Log(Eh)")
    plt.show()

def punto3_tablas_aproximacion_y_graficas():
    f= lambda x, y:-x*y-np.cos(6*x)*exp(-1/2*x**2)
    X=np.linspace(0,1,100)
    solucion=-1/6*exp(-1/2*X**2)*(np.sin(6*X)-6)
    for j in list(range(1,11)):
        cont=0
        todas_x=[]
        todas_y=[]
        tabla=[]
        x0=0
        y0=1
        yprima=f(x0,y0)
        h=2**(-j)
        todas_x.append(x0)
        todas_y.append(y0)
        tabla.append(['i','h','   xi','  Yi', '     Yi\''])
        tabla.append([cont,h,x0,y0,yprima])
        while x0<1:
            x1=x0+h
            y1=(y0+h*-np.cos(6*x1)*exp(-1/2*x1**2))/(1+h*x1)
            todas_x.append(x1)
            todas_y.append(y1)
            yprima=f(x1,y1)
            cont=cont+1
            tabla.append([cont,h,x1,y1,yprima])
            x0=x1
            y0=y1
        for row in tabla:
            print(" {:>20} {:>5} {: >20} {: >20} {: >20}".format(*row))
        plt.plot(todas_x,todas_y,label="exacta")
        plt.plot(X,solucion,label="aproximada")
        plt.legend(loc="upper right")
        plt.xlabel('x'); plt.ylabel("y");plt.title("Solución real vs aproximada con h="+str(h))
        plt.show()

def punto3_error_y_graficas_error():
    f= lambda x, y:-x*y-np.cos(6*x)*exp(-1/2*x**2)
    solucion=lambda x:-1/6*exp(-1/2*x**2)*(np.sin(6*x)-6)
    tabla=[]
    tabla.append(['j','h','Eh','a'])
    cont=0
    todas_las_h=[]
    todos_los_Eh=[]
    for j in list(range(1,11)):
        cont=cont+1
        x0=0
        y0=1
        h=2**(-j)
        todas_las_h.append(h)
        suma_errores=0
        while x0<1:
            x1=x0+h
            y1=(y0+h*-np.cos(6*x1)*exp(-1/2*x1**2))/(1+h*x1)
            m=(y1-y0)/(x1-x0)
            v=lambda x:m*(x-x0)+y0
            integrando=lambda x:abs(solucion(x)-v(x))**2
            integral1,error_aprox=quad(integrando,x0,x1)
            suma_errores=suma_errores+integral1
            x0=x1
            y0=y1
        Eh=math.sqrt(suma_errores)
        todos_los_Eh.append(Eh)
        if cont>1:
            filaAnt=tabla[cont-1]
            h_ant=filaAnt[1]
            Eh_ant=filaAnt[2]
            a=math.log(Eh/Eh_ant)/math.log(h/h_ant)
        else:
            a='n/a'
        tabla.append([cont,h,Eh,a])
    for row in tabla:
        print("{:>17} {:>17} {:>17} {:>17} ".format(*row))
    plt.plot(todas_las_h,todos_los_Eh)
    plt.title('Eh vs h')
    plt.xlabel=("h");plt.ylabel=("Eh")
    plt.show()
    plt.loglog(todas_las_h,todos_los_Eh)
    plt.title('Log(Eh) vs Log(h)')
    plt.xlabel=("Log(h)");plt.ylabel=("Log(Eh)")
    plt.show()
punto2_tablas_aproximacion_y_graficas()
#punto2_error_y_graficas_error()
#punto3_tablas_aproximacion_y_graficas()
#punto3_error_y_graficas_error()


