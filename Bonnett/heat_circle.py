# -*- coding: utf-8 -*-
"""Heat_Circle.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZnJo7RmvgQXe1Txh2VRkMHA52skbX7M6
"""

import numpy as np
import sympy as sym


from sympy import *
from sympy import reduced, expand, Rational,nsimplify,solve,exp,log,cos,sin,pi,var,diff,symbols,sympify,Sum
from sympy.abc import x, y,u,n,k,m,c,t,z,s,r

k,m,n,c= symbols('k m n c', integer=True,positive=True)
import scipy.integrate as integrate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

rho=7

def f0(x):
  y=(pi-x)
  return y
 
def fc(x,n):
  y=f0(x)*cos(n*x)
  return y
 
def fs(x,n):
  y=f0(x)*sin(n*x)
  return y

a00=(1/pi)*sym.integrate(f0(x), (x, -pi, pi))
check=[2,4,16,32]
val=[]
domain=np.linspace(-np.pi, np.pi, num=200)
domai0=np.linspace(-np.pi, np.pi, num=200)

a0=(1/pi)*sym.integrate(f0(x), (x, -pi, pi))
an=(1/(pi*rho**n))*sym.integrate(fc(x,n), (x, -pi, pi))
bn=(1/(pi*rho**n))*sym.integrate(fs(x,n), (x, -pi, pi)) 

print(sym.latex((sym.simplify(a0))))

sum=a0/2
for n in range(1,33):
  Ic=(1/(pi*rho**n))*sym.integrate(fc(x,n), (x, -pi, pi))
  Is=(1/(pi*rho**n))*sym.integrate(fs(x,n), (x, -pi, pi)) 
 
  sum+=Ic*r**n*cos(n*x)+Is*r**n*sin(n*x)
  if n in check:
    val.append(sum)

z=[]
z.append(sym.lambdify([r,x], val[3], 'numpy'))

zl=z[0]



fig = plt.figure()
ax = Axes3D(fig)

rad = np.linspace(0, 1, 200)
azm = np.linspace(-np.pi, np.pi, 200)
rd, th = np.meshgrid(rad, azm)

p=zl(rd,th)

plt.subplot(projection="polar")

surf=plt.pcolormesh(th, rd, p,cmap=cm.coolwarm)
plt.plot(azm, rd, color='k', ls='none') 
plt.grid()

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

y=4/5
t=pi/4
jk=(1/(2*pi))*((1-y**2)/(1+y**2-2*y*cos(t-x)))*(pi-x)

js=sym.lambdify(x, jk, 'numpy')

result = integrate.quad(lambda x: js(x), -pi, pi)


#uu=(1/(2*pi))*sym.integrate(jk*(pi-x), (x, -pi, pi))

print(result)
print(zl(0.8,np.pi/4))
