#Taller 6
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy.integrate import quad
import math
import sympy as sym
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def punto1a():
    z=[]
    x = sym.Symbol('x')
    r = sym.Symbol('r')
    fs = pow(np.pi,2)/3
    for n in list(range(1,20)):
        fs = fs + (4*pow(-1,n)/(pow(2,n)*pow(n,2)))*pow(r,n)*sym.cos(n*x)+(2*pow(-1,n)/(pow(2,n)*n))*pow(r,n)*sym.sin(n*x)
    z.append(sym.lambdify([r,x], fs, 'numpy'))
    zs=z[0]
    fig = plt.figure()
    ax = Axes3D(fig)
    rad = np.linspace(0, 2, 200)
    azm = np.linspace(-np.pi, np.pi, 200)
    rd, th = np.meshgrid(rad, azm)
    p=zs(rd,th)
    plt.subplot(projection="polar")
    surf=plt.pcolormesh(th, rd, p,cmap=cm.coolwarm)
    plt.plot(azm, rd, color='k', ls='none')
    plt.grid()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def punto1b():
    z=[]
    x = sym.Symbol('x')
    r = sym.Symbol('r')
    fs = 225.13/2
    for n in list(range(1,20)):
        fs = fs + (pow(r,n)*pow(-1,n)/(np.exp(2*np.pi)*pow(4,n)*np.pi*pow(pow(n,2)+4,2))*(((2*np.pi)-1)*pow(n,2)+np.exp(4*np.pi)*((2*np.pi+1)*pow(n,2)+8*np.pi-4)+8*np.pi+4)*sym.cos(n*x)-n*(np.pi*(pow(n,2)+4)+np.exp(4*np.pi)*(np.pi*(pow(n,2)+4)-4)+4)*sym.sin(n*x))
    z.append(sym.lambdify([r,x], fs, 'numpy'))
    zs=z[0]
    fig = plt.figure()
    ax = Axes3D(fig)
    rad = np.linspace(0, 4, 200)
    azm = np.linspace(-np.pi, np.pi, 200)
    rd, th = np.meshgrid(rad, azm)
    p=zs(rd,th)
    plt.subplot(projection="polar")
    surf=plt.pcolormesh(th, rd, p,cmap=cm.coolwarm)
    plt.plot(azm, rd, color='k', ls='none')
    plt.grid()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def punto2a():
    z=[]
    x = sym.Symbol('x')
    r = sym.Symbol('r')
    fs = pow(r,2)*sym.cos(2*x)
    z.append(sym.lambdify([r,x], fs, 'numpy'))
    zs=z[0]
    fig = plt.figure()
    ax = Axes3D(fig)
    rad = np.linspace(0, 4, 200)
    azm = np.linspace(-np.pi, np.pi, 200)
    rd, th = np.meshgrid(rad, azm)
    p=zs(rd,th)
    plt.subplot(projection="polar")
    surf=plt.pcolormesh(th, rd, p,cmap=cm.coolwarm)
    plt.plot(azm, rd, color='k', ls='none')
    plt.grid()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def punto2b():
    z=[]
    x = sym.Symbol('x')
    r = sym.Symbol('r')
    fs = (1/2)*pow(r,2)*sym.sin(2*x)
    z.append(sym.lambdify([r,x], fs, 'numpy'))
    zs=z[0]
    fig = plt.figure()
    ax = Axes3D(fig)
    rad = np.linspace(0, 3, 200)
    azm = np.linspace(-np.pi, np.pi, 200)
    rd, th = np.meshgrid(rad, azm)
    p=zs(rd,th)
    plt.subplot(projection="polar")
    surf=plt.pcolormesh(th, rd, p,cmap=cm.coolwarm)
    plt.plot(azm, rd, color='k', ls='none')
    plt.grid()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def punto3_b1_c_d():
    print ('                            Con p=2                             ')
    errores = []
    acum_h = []
    x = 0.5
    fprima_real = lambda x: 0.988*np.cos(x)+0.004*x*np.sin(x)
    valor_real = fprima_real(x)
    f = lambda x: 0.004*x*sym.cos(x)-sym.sin(x)
    h0 = pow(2,-1)
    fprima_aprox = (-f(x-2*h0)+2*f(x-h0)-2*f(x+h0)+f(x+2*h0))/(2*pow(h0,3))
    E0 = abs(fprima_aprox-valor_real)
    errores.append(E0)
    acum_h.append(h0)
    print ('j','2^j','      f\'\'\'(0.5)','           E', '              alpha')
    print(1,2,fprima_aprox,E0,'          -')
    for j in list(range(2,21)):
        h = pow(2,-j)
        fprima_aprox = (-f(x-2*h)+2*f(x-h)-2*f(x+h)+f(x+2*h))/(2*pow(h,3))
        E = abs(fprima_aprox-valor_real)
        alpha = math.log((E)/(E0))/math.log((h)/(h0))
        print(j,pow(2,j),fprima_aprox,E,alpha)
        errores.append(E)
        acum_h.append(h)
        E0 = E
        h0 = h
    plt.figure(1)
    plt.xlabel('h')
    plt.ylabel("E")
    plt.title("E vs h para p=2")
    plt.plot(acum_h,errores)
    plt.tight_layout()
    plt.show()
    plt.figure(2)
    plt.xlabel('Log(h)')
    plt.ylabel("Log(E)")
    plt.title("Log(E) vs Log(h) para p=2")
    plt.loglog(acum_h,errores)
    plt.tight_layout()
    plt.show()

def punto3_b2_c_d():
    print ('                            Con p=4                             ')
    errores = []
    acum_h = []
    x = 0.5
    fprima_real = lambda x: 0.988*np.cos(x)+0.004*x*np.sin(x)
    valor_real = fprima_real(x)
    f = lambda x: 0.004*x*sym.cos(x)-sym.sin(x)
    h0 = pow(2,-1)
    fprima_aprox = (f(x-3*h0)-8*f(x-2*h0)+13*f(x-h0)-13*f(x+h0)+8*f(x+2*h0)-f(x+3*h0))/(8*pow(h0,3))
    E0 = abs(fprima_aprox-valor_real)
    errores.append(E0)
    acum_h.append(h0)
    print ('j','2^j','      f\'\'\'(0.5)','           E', '              alpha')
    print(1,2,fprima_aprox,E0,'          -')
    for j in list(range(2,21)):
        h = pow(2,-j)
        fprima_aprox = (f(x-3*h)-8*f(x-2*h)+13*f(x-h)-13*f(x+h)+8*f(x+2*h)-f(x+3*h))/(8*pow(h,3))
        E = abs(fprima_aprox-valor_real)
        alpha = math.log((E)/(E0))/math.log((h)/(h0))
        print(j,pow(2,j),fprima_aprox,E,alpha)
        errores.append(E)
        acum_h.append(h)
        E0 = E
        h0 = h
    plt.figure(1)
    plt.xlabel('h')
    plt.ylabel("E")
    plt.title("E vs h para p=4")
    plt.plot(acum_h,errores)
    plt.tight_layout()
    plt.show()
    plt.figure(2)
    plt.xlabel('Log(h)')
    plt.ylabel("Log(E)")
    plt.title("Log(E) vs Log(h) para p=4")
    plt.loglog(acum_h,errores)
    plt.tight_layout()
    plt.show()

def punto3_b3_c_d():
    print ('                            Con p=6                             ')
    errores = []
    acum_h = []
    x = 0.5
    fprima_real = lambda x: 0.988*np.cos(x)+0.004*x*np.sin(x)
    valor_real = fprima_real(x)
    f = lambda x: 0.004*x*sym.cos(x)-sym.sin(x)
    h0 = pow(2,-1)
    fprima_aprox = (-7*f(x-4*h0)+72*f(x-3*h0)-338*f(x-2*h0)+488*f(x-h0)-488*f(x+h0)+338*f(x+2*h0)-72*f(x+3*h0)+7*f(x+4*h0))/(240*pow(h0,3))
    E0 = abs(fprima_aprox-valor_real)
    errores.append(E0)
    acum_h.append(h0)
    print ('j','2^j','      f\'\'\'(0.5)','           E', '              alpha')
    print(1,2,fprima_aprox,E0,'          -')
    for j in list(range(2,21)):
        h = pow(2,-j)
        fprima_aprox = (-7*f(x-4*h)+72*f(x-3*h)-338*f(x-2*h)+488*f(x-h)-488*f(x+h)+338*f(x+2*h)-72*f(x+3*h)+7*f(x+4*h))/(240*pow(h,3))
        E = abs(fprima_aprox-valor_real)
        alpha = math.log((E)/(E0))/math.log((h)/(h0))
        print(j,pow(2,j),fprima_aprox,E,alpha)
        errores.append(E)
        acum_h.append(h)
        E0 = E
        h0 = h
    plt.figure(1)
    plt.xlabel('h')
    plt.ylabel("E")
    plt.title("E vs h para p=6")
    plt.plot(acum_h,errores)
    plt.tight_layout()
    plt.show()
    plt.figure(2)
    plt.xlabel('Log(h)')
    plt.ylabel("Log(E)")
    plt.title("Log(E) vs Log(h) para p=6")
    plt.loglog(acum_h,errores)
    plt.tight_layout()
    plt.show()

def punto3_c_djuntos():
    errores1 = []
    errores2 = []
    errores3 = []
    acum_h = []
    x = 0.5
    fprima_real = lambda x: 0.988*np.cos(x)+0.004*x*np.sin(x)
    valor_real = fprima_real(x)
    f = lambda x: 0.004*x*sym.cos(x)-sym.sin(x)
    h0 = pow(2,-1)
    fprima_aprox1 = (-f(x-2*h0)+2*f(x-h0)-2*f(x+h0)+f(x+2*h0))/(2*pow(h0,3))
    fprima_aprox2 = (f(x-3*h0)-8*f(x-2*h0)+13*f(x-h0)-13*f(x+h0)+8*f(x+2*h0)-f(x+3*h0))/(8*pow(h0,3))
    fprima_aprox3 = (-7*f(x-4*h0)+72*f(x-3*h0)-338*f(x-2*h0)+488*f(x-h0)-488*f(x+h0)+338*f(x+2*h0)-72*f(x+3*h0)+7*f(x+4*h0))/(240*pow(h0,3))
    E1 = abs(fprima_aprox1-valor_real)
    E2 = abs(fprima_aprox2-valor_real)
    E3 = abs(fprima_aprox3-valor_real)
    errores1.append(E1)
    errores2.append(E2)
    errores3.append(E3)
    acum_h.append(h0)
    for j in list(range(2,21)):
        h = pow(2,-j)
        fprima_aprox1 = (-f(x-2*h)+2*f(x-h)-2*f(x+h)+f(x+2*h))/(2*pow(h,3))
        fprima_aprox2 = (f(x-3*h)-8*f(x-2*h)+13*f(x-h)-13*f(x+h)+8*f(x+2*h)-f(x+3*h))/(8*pow(h,3))
        fprima_aprox3 = (-7*f(x-4*h)+72*f(x-3*h)-338*f(x-2*h)+488*f(x-h)-488*f(x+h)+338*f(x+2*h)-72*f(x+3*h)+7*f(x+4*h))/(240*pow(h,3))
        E1 = abs(fprima_aprox1-valor_real)
        E2 = abs(fprima_aprox2-valor_real)
        E3 = abs(fprima_aprox3-valor_real)
        errores1.append(E1)
        errores2.append(E2)
        errores3.append(E3)
        acum_h.append(h)
    plt.figure(1)
    plt.plot(acum_h,errores1,label="p=2")
    plt.plot(acum_h,errores2,label="p=4")
    plt.plot(acum_h,errores3,label="p=6")
    plt.legend(loc="upper right")
    plt.xlabel('h')
    plt.ylabel("E")
    plt.show()
    plt.figure(2)
    plt.loglog(acum_h,errores1,label="p=2")
    plt.loglog(acum_h,errores2,label="p=4")
    plt.loglog(acum_h,errores3,label="p=6")
    plt.legend(loc="upper right")
    plt.xlabel('Log(h)')
    plt.ylabel("Log(E)")
    plt.show()

#punto1a()
#punto1b()
#punto2a()
#punto2b()
#punto3_b1_c_d()
#punto3_b2_c_d()
#punto3_b3_c_d()
punto3_c_djuntos()


