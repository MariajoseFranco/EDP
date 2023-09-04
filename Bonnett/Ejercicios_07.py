import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fminbound
from sympy import *

k,m,n,c= symbols('k m n c', integer=True,positive=True)


class Ejercicios07:

    def punto_1a(self):
        # #Cálculo de los coeficientes con grado de precisión p=1
        # a = np.array([[1, 1], [0, 1]])
        # b = np.array([0,1])
        # x = np.linalg.solve(a,b)
        # print('Los coeficientes para d=1 y p=1 son: ', x)

        # #Cálculo de los coeficientes con grado de precisión p=2
        # a = np.array([[1, 1, 1], [0, 1, 2], [0, 1, 4]])
        # b = np.array([0,1,0])
        # x = np.linalg.solve(a,b)

        # print('Los coeficientes para d=1 y p=2 son: ', x)

        # #Cálculo de los coeficientes con grado de precisión p=3
        # a = np.array([[1, 1, 1, 1], [0, 1, 2, 3], [0, 1, 4, 9], [0, 1, 8, 27]])
        # b = np.array([0, 1, 0, 0])
        # x = np.linalg.solve(a,b)

        # print('Los coeficientes para d=1 y p=3 son: ', x)


        # #Cálculo de los coeficientes con grado de precisión p=4
        # a = np.array([[1, 1, 1, 1, 1], [0, 1, 2, 3, 4], [0, 1, 4, 9, 16], [0, 1, 8, 27, 64], [0, 1, 16, 81, 256]])
        # b = np.array([0, 1, 0, 0, 0])
        # x = np.linalg.solve(a,b)
        # print('Los coeficientes para d=1 y p=4 son: ', x)

        # #Cálculo de los coeficientes con grado de precisión p=6
        # a = np.array([[1, 1, 1, 1, 1, 1, 7], [0, 1, 2, 3, 4, 5, 6], [0, 1, 4, 9, 16, 25, 36], [0, 1, 8, 27, 64, 125, 216], [0, 1, 16, 81, 256, 625, 1296], [0, 1, 32, 243, 1024, 3125, 7776], [0, 1, 64, 729, 4096, 15625, 46656]])
        # b = np.array([0, 1, 0, 0, 0, 0, 0])
        # x = np.linalg.solve(a,b)
        # print('Los coeficientes para d=1 y p=6 son: ', x)

        # #Cálculo de los coeficientes con grado de precisión p=1
        # a = np.array([[1, 1], [0, 1]])
        # b = np.array([0, 1])
        # x = np.linalg.solve(a,b)
        # print('Los coeficientes para d=1 y p=1 son: ', x)

        # #Cálculo de los coeficientes con grado de precisión p=2
        # a = np.array([[1, 1, 1], [-2, -1, 0], [4, 1, 0]])
        # b = np.array([0, 1, 0])
        # x = np.linalg.solve(a,b)
        # print('Los coeficientes para d=1 y p=2 son: ', x)

        # #Cálculo de los coeficientes con grado de precisión p=3
        # a = np.array([[1, 1, 1, 1], [-3, -2, -1, 0], [9, 4, 1, 0], [-27, -8, -1, 0]])
        # b = np.array([0, 1, 0, 0])
        # x = np.linalg.solve(a,b)
        # print('Los coeficientes para d=1 y p=3 son: ', x)

        # #Cálculo de los coeficientes con grado de precisión p=4
        # a = np.array([[1, 1, 1, 1, 1], [-4, -3, -2, -1, 0], [16, 9, 4, 1, 0], [-64, -27, -8, -1, 0], [256, 81, 16, 1, 0]])
        # b = np.array([0, 1, 0, 0, 0])
        # x = np.linalg.solve(a,b)
        # print('Los coeficientes para d=1 y p=4 son: ', x)

        #Cálculo de los coeficientes con grado de precisión p=6
        a = np.array([[1, 1, 1, 1, 1, 1, 1], [-6, -5, -4, -3, -2, -1, 0], [36, 25, 16, 9, 4, 1, 0], [-216, -125, -64, -27, -8, -1, 0], [1296, 625, 256, 81, 16, 1, 0], [-7776, -3125, -1024, -243, -32, -1, 0], [46656, 15625, 4096, 729, 64, 1, 0]])
        b = np.array([0, 0, 0, 1, 0, 0, 0])
        x = np.linalg.solve(a,b)
        print('Los coeficientes para d=1 y p=6 son: ', x)

    def punto_1b(self):
        a = np.pi/2
        #Cálculo de la tabla para un grado de precisión p=1
        df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        f_1x_approximate = lambda x, h: (f_x(x + h) - f_x(x))/(h)

        for j in range(1, 21):
            h = 2**(-j)
            df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        for j in range(1, 20):
            numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
            denominator = np.log(float(2**(-j)/2**(-j + 1)))
            df.iloc[j]['alpha_h']  = numerator/denominator

        h = [2**(-j) for j in range(1, 21)]
        plt.plot(h, np.array(df['E']))
        plt.title('Eh vs h for p=1')
        plt.show()
        plt.loglog(h , np.array(df['E']))
        plt.title('Log(Eh) vs Log(h) for p=1')
        plt.show()

        print(df)

        #Cálculo de la tabla para un grado de precisión p=2
        df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        f_1x_approximate = lambda x, h: (4*f_x(x + h) - 3*f_x(a) - f_x(x+2*h))/(2*h)

        for j in range(1, 21):
            h = 2**(-j)
            df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        for j in range(1, 20):
            numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
            denominator = np.log(float(2**(-j)/2**(-j + 1)))
            df.iloc[j]['alpha_h']  = numerator/denominator

        h = [2**(-j) for j in range(1, 21)]
        plt.plot(h, np.array(df['E']))
        plt.title('Eh vs h for p=2')
        plt.show()
        plt.loglog(h , np.array(df['E']))
        plt.title('Log(Eh) vs Log(h) for p=2')
        plt.show()
        print(df)

        #Cálculo de la tabla para un grado de precisión p=3
        df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        f_1x_approximate = lambda x, h: (-11*f_x(x) + 18*f_x(x + h) - 9*f_x(x + 2*h) + 2*f_x(x + 3*h))/(6*h)

        for j in range(1, 21):
            h = 2**(-j)
            df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        for j in range(1, 20):
            numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
            denominator = np.log(float(2**(-j)/2**(-j + 1)))
            df.iloc[j]['alpha_h']  = numerator/denominator

        h = [2**(-j) for j in range(1, 21)]
        plt.plot(h, np.array(df['E']))
        plt.title('Eh vs h for p=3')
        plt.show()
        plt.loglog(h , np.array(df['E']))
        plt.title('Log(Eh) vs Log(h) for p=3')
        plt.show()
        print(df)

        #Cálculo de la tabla para un grado de precisión p=4
        df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        f_1x_approximate = lambda x, h: (-25*f_x(x) + 48*f_x(x + h) - 36*f_x(x + 2*h) + 16*f_x(x + 3*h) - 3*f_x(x + 4*h))/(12*h)

        for j in range(1, 21):
            h = 2**(-j)
            df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        for j in range(1, 20):
            numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
            denominator = np.log(float(2**(-j)/2**(-j + 1)))
            df.iloc[j]['alpha_h']  = numerator/denominator

        h = [2**(-j) for j in range(1, 21)]
        plt.plot(h, np.array(df['E']))
        plt.title('Eh vs h for p=4')
        plt.show()
        plt.loglog(h , np.array(df['E']))
        plt.title('Log(Eh) vs Log(h) for p=4')
        plt.show()
        print(df)


        #Cálculo de la tabla para un grado de precisión p=6
        df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        f_1x_approximate = lambda x, h: (-87*f_x(x) + 360*f_x(x + h) - 450*f_x(x + 2*h) + 400*f_x(x + 3*h) - 225*f_x(x + 4*h) + 72*f_x(x + 5*h) -10*f_x(x + 6*h))/(60*h)

        for j in range(1, 21):
            h = 2**(-j)
            df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        for j in range(1, 20):
            numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
            denominator = np.log(float(2**(-j)/2**(-j + 1)))
            df.iloc[j]['alpha_h']  = numerator/denominator

        h = [2**(-j) for j in range(1, 21)]
        plt.plot(h, np.array(df['E']))
        plt.title('Eh vs h for p=6')
        plt.show()
        plt.loglog(h , np.array(df['E']))
        plt.title('Log(Eh) vs Log(h) for p=6')
        plt.show()
        print(df)

        # a = np.pi/2
        # #Cálculo de la tabla para un grado de precisión p=1
        # df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        # f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        # f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        # f_1x_approximate = lambda x, h: (f_x(x) - f_x(x - h))/(h)

        # for j in range(1, 21):
        #     h = 2**(-j)
        #     df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        # for j in range(1, 20):
        #     numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
        #     denominator = np.log(float(2**(-j)/2**(-j + 1)))
        #     df.iloc[j]['alpha_h']  = numerator/denominator

        # h = [2**(-j) for j in range(1, 21)]
        # plt.plot(h, np.array(df['E']))
        # plt.title('Eh vs h for p=1')
        # plt.show()
        # plt.loglog(h , np.array(df['E']))
        # plt.title('Log(Eh) vs Log(h) for p=1')
        # plt.show()

        # print(df)

        # #Cálculo de la tabla para un grado de precisión p=2
        # df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        # f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        # f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        # f_1x_approximate = lambda x, h: (3*f_x(x) - 4*f_x(x - h) + f_x(x - 2*h))/(2*h)

        # for j in range(1, 21):
        #     h = 2**(-j)
        #     df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        # for j in range(1, 20):
        #     numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
        #     denominator = np.log(float(2**(-j)/2**(-j + 1)))
        #     df.iloc[j]['alpha_h']  = numerator/denominator

        # h = [2**(-j) for j in range(1, 21)]
        # plt.plot(h, np.array(df['E']))
        # plt.title('Eh vs h for p=2')
        # plt.show()
        # plt.loglog(h , np.array(df['E']))
        # plt.title('Log(Eh) vs Log(h) for p=2')
        # plt.show()
        # print(df)

        # #Cálculo de la tabla para un grado de precisión p=3
        # df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        # f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        # f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        # f_1x_approximate = lambda x, h: (11*f_x(x) - 18*f_x(x - h) + 9*f_x(x - 2*h) - 2*f_x(x - 3*h))/(6*h)

        # for j in range(1, 21):
        #     h = 2**(-j)
        #     df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        # for j in range(1, 20):
        #     numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
        #     denominator = np.log(float(2**(-j)/2**(-j + 1)))
        #     df.iloc[j]['alpha_h']  = numerator/denominator

        # h = [2**(-j) for j in range(1, 21)]
        # plt.plot(h, np.array(df['E']))
        # plt.title('Eh vs h for p=3')
        # plt.show()
        # plt.loglog(h , np.array(df['E']))
        # plt.title('Log(Eh) vs Log(h) for p=3')
        # plt.show()
        # print(df)

        # #Cálculo de la tabla para un grado de precisión p=4
        # df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        # f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        # f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        # f_1x_approximate = lambda x, h: (25*f_x(x) - 48*f_x(x - h) + 36*f_x(x - 2*h) - 16*f_x(x - 3*h) + 3*f_x(x - 4*h))/(12*h)

        # for j in range(1, 21):
        #     h = 2**(-j)
        #     df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        # for j in range(1, 20):
        #     numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
        #     denominator = np.log(float(2**(-j)/2**(-j + 1)))
        #     df.iloc[j]['alpha_h']  = numerator/denominator

        # h = [2**(-j) for j in range(1, 21)]
        # plt.plot(h, np.array(df['E']))
        # plt.title('Eh vs h for p=4')
        # plt.show()
        # plt.loglog(h , np.array(df['E']))
        # plt.title('Log(Eh) vs Log(h) for p=4')
        # plt.show()
        # print(df)


        # #Cálculo de la tabla para un grado de precisión p=6
        # df = pd.DataFrame(columns = ['j', '2**j', "f(2)x", 'E', 'alpha_h'], index = range(1,21))

        # f_x              = lambda x: x**2*np.sin(x) - x*np.cos(x**2)
        # f_1x             = lambda x: 2*x*np.sin(x) + x**2 * np.cos(x) - cos(x**2) + 2*x**2 * np.sin(x**2)
        # f_1x_approximate = lambda x, h: (87*f_x(x) - 360*f_x(x - h) + 450*f_x(x - 2*h) - 400*f_x(x - 3*h) + 225*f_x(x - 4*h) - 72*f_x(x - 5*h) + 10*f_x(x - 6*h))/(60*h)

        # for j in range(1, 21):
        #     h = 2**(-j)
        #     df.loc[j] = [j, 2**j, f_1x_approximate(a, h), abs(f_1x(a) - f_1x_approximate(a, h)), -1]

        # for j in range(1, 20):
        #     numerator   = np.log(float(df.iloc[j]['E']/df.iloc[j - 1]['E']))
        #     denominator = np.log(float(2**(-j)/2**(-j + 1)))
        #     df.iloc[j]['alpha_h']  = numerator/denominator

        # h = [2**(-j) for j in range(1, 21)]
        # plt.plot(h, np.array(df['E']))
        # plt.title('Eh vs h for p=6')
        # plt.show()
        # plt.loglog(h , np.array(df['E']))
        # plt.title('Log(Eh) vs Log(h) for p=6')
        # plt.show()
        # print(df)

    def punto_1f(self):
        a   = np.pi/2
        x = Symbol('x')
        h = symbols('h')
        epsilon = 2.2*10**(-16)
        error_h = 1*10**(-16)
        
        f_x = (x**2)*sin(x) - x*cos(x**2)
        f_1x = f_x.diff(x)
        f_2x = f_1x.diff(x)
        f_3x = f_2x.diff(x)
        f_4x = f_3x.diff(x)
        f_5x = f_4x.diff(x)
        f_6x = f_5x.diff(x)
        f_7x = f_6x.diff(x)

        f = lambdify(x, f_x, 'numpy')
        f1 = lambdify(x, - f_1x, 'numpy')
        f2 = lambdify(x, - f_2x, 'numpy')
        f3 = lambdify(x, - f_3x, 'numpy')
        f4 = lambdify(x, - f_4x, 'numpy')
        f5 = lambdify(x, - f_5x, 'numpy')
        f6 = lambdify(x, - f_6x, 'numpy')
        f7 = lambdify(x, - f_7x, 'numpy')

        # c = fminbound(f2, a, a + 0.5)/np.math.factorial(2)
        # c = fminbound(f3, a, a + 0.5)/np.math.factorial(3)
        # c = fminbound(f4, a, a + 0.5)/np.math.factorial(4)
        # c = fminbound(f5, a, a + 0.5)/np.math.factorial(5)
        # c = fminbound(f7, a, a + 0.5)/np.math.factorial(7)

        # c = fminbound(f2, a - 0.5, a)/np.math.factorial(2)
        # c = fminbound(f3, a - 0.5, a)/np.math.factorial(3)
        # c = fminbound(f4, a - 0.5, a)/np.math.factorial(4)
        # c = fminbound(f5, a - 0.5, a)/np.math.factorial(5)
        c = fminbound(f7, a - 0.5, a)/np.math.factorial(7)

        # print(c)

        # print(solve(c*h + (epsilon/h) - error_h, h))
        # print(np.sqrt(epsilon/c))

        # print(solve(c*h**2 + (epsilon/h) - error_h, h))
        # print((epsilon/2*c)**(1/3))

        # print(solve(c*h**3 + (epsilon/h) - error_h, h))
        # print((epsilon/3*c)**(1/4))

        # print(solve(c*h**4 + (epsilon/h) - error_h, h))
        # print((epsilon/4*c)**(1/5))

        # print(solve(c*h**3 + (epsilon/h) - error_h, h))
        # print((epsilon/6*c)**(1/7))


        # print(solve(c*h + (epsilon/h) - error_h, h))
        # print(np.sqrt(epsilon/c))

        # print(solve(c*h**2 + (epsilon/h) - error_h, h))
        # print((epsilon/2*c)**(1/3))

        # print(solve(c*h**3 + (epsilon/h) - error_h, h))
        # print((epsilon/3*c)**(1/4))

        # print(solve(c*h**4 + (epsilon/h) - error_h, h))
        # print((epsilon/4*c)**(1/5))

        print(solve(c*h**3 + (epsilon/h) - error_h, h))
        print((epsilon/6*c)**(1/7))

    def punto_3(self):
        a = 0
        b = 1
        df_errores = pd.DataFrame(columns = ['h', 'Eh', 'alpha_h'])
        for i in range(1, 11):
            h = 2**(-i)
            x0 = 0
            Y0 = -1
            f_x  = lambda x, y: (1/10)*(np.exp(x/10)*(y-x)**2) + 1

            df = pd.DataFrame(columns = ['j', 'Xj', "Yj", "Y'j", "Eh", "h"], index = range(int((b-a)/h) + 1))
            df.loc[0] = [0, x0, Y0, f_x(x0, Y0), 0, h]
            for j in range(1, int((b-a)/h) + 1):
                Xj = a + j*h
                Yj = df['Yj'][j - 1] + h*f_x(df["Xj"][j - 1], df["Yj"][j - 1])
                Y1j = f_x(Xj, Yj)
                df.loc[j] = [j, Xj, Yj, Y1j, 0, h]

            for j in range(int((b-a)/h)):
            
                x = Symbol('x')
                x0 = df["Xj"][j]
                x1 = df["Xj"][j + 1]
                integral = abs(x - exp(-x/10) - ((df["Yj"][j] - df["Yj"][j + 1])/(df["Xj"][j] - df["Xj"][j + 1]) * (x - x0) + df["Yj"][j]) ) **2
                js = lambdify(x, integral, 'numpy')
                df["Eh"][j] = quad(lambda x: js(x), x0, x1)[0]

            
            
            # print(f"Sum of errors for h = {h} is {np.sqrt(sum(df['Eh']))}" )
            # print('-------------------------------------------------------------------')
            df_errores.loc[h] = [h, np.sqrt(sum(df['Eh'])), 0]

        for j in range(1, 10):
            numerator   = np.log(float(df_errores.iloc[j]['Eh']/df_errores.iloc[j - 1]['Eh']))
            denominator = np.log(float(2**(-j)/2**(-j + 1)))
            df_errores.iloc[j]['alpha_h']  = numerator/denominator
            # x = np.arange(0,1,0.000001)
            # y = x - np.exp(-x/10)
        
        print(df_errores)

        # plt.plot(df_errores['h'],df_errores['Eh'])
        # plt.title('Graph of Eh versus h')
        # plt.legend(['Real', 'Predicted'])
        # plt.show()
        plt.loglog(df_errores['h'] , df_errores['Eh'])
        plt.title('Graph of log(Eh) versus log(h)')
        plt.show()
            # print(df) 
            # plt.plot(x,y)
            # plt.plot(df['Xj'], df['Yj'])
            # plt.title(f'Graph of Y versus Y_hat with j = {i}')
            # plt.legend(['Real', 'Predicted'])
            # plt.show()
            # print(df) 
        
if __name__ == "__main__":
    ejercicio = Ejercicios07()
    ejercicio.punto_1a()