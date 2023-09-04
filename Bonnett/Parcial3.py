import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.integrate import quad
from sympy import *

k,m,n,c= symbols('k m n c', integer=True,positive=True)


class Parcial3:

    def punto_1(self, display_content, graph):
        r, theta = symbols('r theta')
        n = symbols('n', integer=True, positive=True)
        rho = 1

        f = 7 + 9*cos(theta)
        

        a0 = (1/pi) * integrate(f, (theta, -pi, pi))
        an = (1/(pi*(rho**n))) * integrate(f * cos(n*theta), (theta, -pi, pi))
        bn = (1/(pi*(rho**n))) * integrate(f * sin(n*theta), (theta, -pi, pi))
        u = (1/2) + ((r**2)/98) * cos(2*theta)
        

        if display_content:
            print('La funcion es: ', simplify(f))
            print('El coeficiente a0 es: ', simplify(a0))
            pprint(simplify(an))
            print('El coeficiente bn es: ', simplify(bn))
            print('La función solución es: ', simplify(u))

        if graph:
            domain_radius = np.linspace(0, rho, 200)
            domain_theta = np.linspace(-np.pi, np.pi, 200)
            rd, th = np.meshgrid(domain_radius, domain_theta)
            y = lambdify([r, theta], u, 'numpy')

            plt.subplots(figsize=(8, 8))
            plt.subplot(projection="polar")
            plt.pcolormesh(th, rd, y(rd,th), cmap=cm.coolwarm)
            plt.grid()
            plt.show()

if __name__ == "__main__":
    ejercicio = Parcial3()
    ejercicio.punto_1(True, False)
