import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.integrate import quad
from sympy import *

k,m,n,c= symbols('k m n c', integer=True,positive=True)


class Ejercicios06:

    def punto_1a(self, display_content, graph):
        r, theta = symbols('r theta')
        n = symbols('n', integer=True, positive=True)
        rho = 7

        f = cos(theta)**2
        

        a0 = (1/pi) * integrate(f, (theta, -pi, pi))
        an = (1/(pi*(rho**n))) * integrate(f * cos(n*theta), (theta, -pi, pi))
        bn = (1/(pi*(rho**n))) * integrate(f * sin(n*theta), (theta, -pi, pi))
        u = (1/2) + ((r**2)/98) * cos(2*theta)
        

        if display_content:
            print('La funcion es: ', simplify(f))
            print('El coeficiente a0 es: ', simplify(a0))
            print('El coeficiente an es: ', simplify(an))
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

    def punto_1b(self, display_content, graph):
        r, theta = symbols('r theta')
        n = symbols('n', integer=True, positive=True)
        rho = 6

        f = sin(theta)**3 + cos(theta)**2
        

        a0 = (1/pi) * integrate(f, (theta, -pi, pi))
        an = (1/(pi*(rho**n))) * integrate(f * cos(n*theta), (theta, -pi, pi))
        bn = (1/(pi*(rho**n))) * integrate(f * sin(n*theta), (theta, -pi, pi))
        u  = (1/2) + ((r**2)/72) * cos(2*theta) + ((r/8) * sin(theta)) - ((r**3)/864) * sin(3*theta)
        

        if display_content:
            print('La funcion es: ', simplify(f))
            print('El coeficiente a0 es: ', simplify(a0))
            print('El coeficiente an es: ', simplify(an))
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

    def punto_2a(self, display_content, graph):
        r, theta = symbols('r theta')
        n = symbols('n', integer=True, positive=True)
        rho = 4

        f = 256 * cos(theta)**2 * sin(theta)**2
        

        a0 = (1/pi) * integrate(f, (theta, -pi, pi))
        an = (1/(pi*(rho**n))) * integrate(f * cos(n*theta), (theta, -pi, pi))
        bn = (1/(pi*(rho**n))) * integrate(f * sin(n*theta), (theta, -pi, pi))
        u  = 32 - 1/8* r**4 * cos(4*theta)
        

        if display_content:
            print('La funcion es: ', simplify(f))
            print('El coeficiente a0 es: ', simplify(a0))
            print('El coeficiente an es: ', simplify(an))
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

        s = Symbol('s')
        r = 3
        theta = pi/2
        integral = (1/(2*pi)) * ((rho**2 - r**2)/(rho**2 + r**2 - 2*rho*r*cos(theta - s))) * 256 * cos(theta)**2 * sin(theta)**2
        print(integral)
        js = lambdify(s, integral, 'numpy')
        print(quad(lambda s: js(s), -pi, pi))

    def punto_2b(self, display_content, graph):
        r, theta = symbols('r theta')
        n = symbols('n', integer=True, positive=True)
        rho = 3

        f = 3 * (cos(theta) - sin(theta))
        

        a0 = (1/pi) * integrate(f, (theta, -pi, pi))
        an = (1/(pi*(rho**n))) * integrate(f * cos(n*theta), (theta, -pi, pi))
        bn = (1/(pi*(rho**n))) * integrate(f * sin(n*theta), (theta, -pi, pi))
        u  = r*cos(theta) - r*sin(theta)
        

        if display_content:
            print('La funcion es: ', simplify(f))
            print('El coeficiente a0 es: ', simplify(a0))
            print('El coeficiente an es: ', simplify(an))
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

        s = Symbol('s')
        r = 3
        theta = pi/2
        integral = (1/(2*pi)) * ((rho**2 - r**2)/(rho**2 + r**2 - 2*rho*r*cos(theta - s))) * 3*(cos(s) - sin(s))
        print(integral)
        js = lambdify(s, integral, 'numpy')
        print(quad(lambda s: js(s), -pi, pi))


    def punto_3a(self):
        #Cálculo de los coeficientes con grado de precisión p=2
        a = np.array([[1, 1, 1], [-1, 0, 1], [1, 0, 1]])
        b = np.array([0,0,1])
        x = np.linalg.solve(a,b)

        print('Los coeficientes para d=2 y p=2 son: ', x)

        #Cálculo de los coeficientes con grado de precisión p=4
        a = np.array([[1, 1, 1, 1, 1], [-2, -1, 0, 1, 2], [4, 1, 0, 1, 4], [-8, -1, 0, 1, 8], [16, 1, 0, 1, 16]])
        b = np.array([0,0,1,0,0])
        x = np.linalg.solve(a,b)

        print('Los coeficientes para d=2 y p=4 son: ', x)


        #Cálculo de los coeficientes con grado de precisión p=6
        a = np.array([[1, 1, 1, 1, 1, 1, 1], [-3, -2, -1, 0 , 1, 2, 3], [9, 4, 1, 0, 1, 4, 9], [-27, -8, -1, 0, 1, 8, 27], [81, 16, 1, 0, 1, 16, 81], [-243, -32, -1, 0, 1, 32, 243], [729, 64, 1, 0, 1, 64, 729]])
        b = np.array([0,0,1,0,0,0,0])
        x = np.linalg.solve(a,b)

        print('Los coeficientes para d=2 y p=6 son: ', x)

    def punto_3b(self):
        #Cálculo de la tabla para un grado de precisión p=2
        df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        f_x              = lambda x: 2*x*np.sin(x) - np.cos(x)
        f_2x             = lambda x: 5*np.cos(x) - 2*x*np.sin(x)
        f_2x_approximate = lambda x, h: (f_x(0.5 + h) - 2*f_x(0.5) + f_x(0.5 - h))/(h**2)

        for j in range(1, 21):
            h = 2**(-j)
            df.loc[j] = [j, 2**j, f_2x_approximate(0.5, h), abs(f_2x(0.5) - f_2x_approximate(0.5, h)), -1]

        for j in range(1, 20):
            if j > 1:
                numerator   = np.log(df.iloc[j]['E']/df.iloc[j - 1]['E'])
                denominator = np.log(2**(-j)/2**(-j + 1))
                df.iloc[j]['alpha_h']  = numerator/denominator

        h = [2**(-j) for j in range(1, 21)]
        plt.plot(h, np.array(df['E']))
        plt.title('Eh vs h for p=2')
        plt.show()
        plt.loglog(h , np.array(df['E']))
        plt.title('Log(Eh) vs Log(h) for p=2')
        plt.show()

        #Cálculo de la tabla para un grado de precisión p=4
        df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        f_x              = lambda x: 2*x*np.sin(x) - np.cos(x)
        f_2x             = lambda x: 5*np.cos(x) - 2*x*np.sin(x)
        f_2x_approximate = lambda x, h: (- 1/12 * f_x(0.5 + 2*h) + 4/3 * f_x(0.5 + h) - 5/2 * f_x(0.5) + 4/3 * f_x(0.5 - h) - 1/12 * f_x(0.5 - 2*h))/(h**2)

        for j in range(1, 21):
            h = 2**(-j)
            df.loc[j] = [j, 2**j, f_2x_approximate(0.5, h), abs(f_2x(0.5) - f_2x_approximate(0.5, h)), -1]

        for j in range(1, 20):
            if j > 1:
                numerator   = np.log(df.iloc[j]['E']/df.iloc[j - 1]['E'])
                denominator = np.log(2**(-j)/2**(-j + 1))
                df.iloc[j]['alpha_h']  = numerator/denominator

        h = [2**(-j) for j in range(1, 21)]
        plt.plot(h, np.array(df['E']))
        plt.title('Eh vs h for p=4')
        plt.show()
        plt.loglog(h , np.array(df['E']))
        plt.title('Log(Eh) vs Log(h) for p=4')
        plt.show()

        #Cálculo de la tabla para un grado de precisión p=6
        df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        f_x              = lambda x: 2*x*np.sin(x) - np.cos(x)
        f_2x             = lambda x: 5*np.cos(x) - 2*x*np.sin(x)
        f_2x_approximate = lambda x, h: (1/90 * f_x(0.5 + 3*h) - 3/20 * f_x(0.5 + 2*h) + 3/2 * f_x(0.5 + h) - 49/18 * f_x(0.5) + 3/2 * f_x(0.5 - h) - 3/20 * f_x(0.5 - 2*h) + 1/90 * f_x(0.5 - 3*h))/(h**2)

        for j in range(1, 21):
            h = 2**(-j)
            df.loc[j] = [j, 2**j, f_2x_approximate(0.5, h), abs(f_2x(0.5) - f_2x_approximate(0.5, h)), -1]

        for j in range(1, 20):
            if j > 1:
                numerator   = np.log(df.iloc[j]['E']/df.iloc[j - 1]['E'])
                denominator = np.log(2**(-j)/2**(-j + 1))
                df.iloc[j]['alpha_h']  = numerator/denominator

        h = [2**(-j) for j in range(1, 21)]
        plt.plot(h, np.array(df['E']))
        plt.title('Eh vs h for p=6')
        plt.show()
        plt.loglog(h , np.array(df['E']))
        plt.title('Log(Eh) vs Log(h) for p=6')
        plt.show()



if __name__ == "__main__":
    ejercicio = Ejercicios06()
    ejercicio.punto_3b()
