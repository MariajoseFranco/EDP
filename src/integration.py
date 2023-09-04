from scipy.integrate import quad
import numpy as np
import sympy as sp

#Definir funcion
'''for n in range(1,7):
    for m in range(1,7):
        f1 = lambda x: (1/(np.pi))*np.cos(n*x)*np.cos(m*x)'''

f1 = lambda x: (1/sp.sqrt(2*sp.pi))*((2*x/sp.pi)+2)
f2 = lambda x: (1/sp.sqrt(2*sp.pi))*2

#Integrar
integ1 = quad(f1,-sp.pi,0)
integ2 = quad(f2,0,sp.pi)
ans1 = integ1[0]
ans2 = integ2[0]

ans = ans1+ans2

#print(ans1)
#print(ans2)
print('La integral es',ans)