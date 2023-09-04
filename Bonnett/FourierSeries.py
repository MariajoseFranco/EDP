# MATEO BONNETT Y VALENTINA YUSTY
import numpy as np
import matplotlib.pyplot as plt

#a = [2,5,10,20,40]
a = [20] #Define all n values
lab = ['f(x)', 'f(x)', 'n=2', 'n=5', 'n=8', 'n=12', 'n=25', 'n=50' ] #Labels for the plot


#Point 4

# x0 = np.arange(-3, 0, 0.01) #First part of f(x)
# y0 = [0 for i in range(300)]

# x1 = np.arange(0, 3, 0.01) #Second part of f(x)
# y1 = [1 for i in range(300)]

# plt.plot(x0,y0, color = 'red')
# plt.plot(x1,y1, color = 'red')

# x0 = np.arange(-9, -6, 0.01) #First part of f(x)
# y0 = [0 for i in range(300)]

# x1 = np.arange(-6, -3, 0.01) #Second part of f(x)
# y1 = [1 for i in range(300)]

# plt.plot(x0,y0, color = 'red')
# plt.plot(x1,y1, color = 'red')

# x0 = np.arange(3, 6, 0.01) #First part of f(x)
# y0 = [0 for i in range(300)]

# x1 = np.arange(6, 9, 0.01) #Second part of f(x)
# y1 = [1 for i in range(300)]

# plt.plot(x0,y0, color = 'red')
# plt.plot(x1,y1, color = 'red')

# plt.scatter(0,0.5, color = 'green')

#Point 5
x0 = np.arange(0, 1, 0.01)
y0 = np.cos(np.pi*x0)

x1 = np.arange(1, 2, 0.01)
y1 = [0 for i in range(100)]

plt.plot(x0,y0, color = 'red')
plt.plot(x1,y1, color = 'red')

x0 = np.arange(2, 3, 0.01)
y0 = np.cos(np.pi*x0)

x1 = np.arange(3, 4, 0.01) 
y1 = [0 for i in range(100)]

plt.plot(x0,y0, color = 'red')
plt.plot(x1,y1, color = 'red')

x0 = np.arange(-2, -1, 0.01) 
y0 = np.cos(np.pi*x0)

x1 = np.arange(-1, 0, 0.01)
y1 = [0 for i in range(100)]

plt.plot(x0,y0, color = 'red')
plt.plot(x1,y1, color = 'red')

plt.scatter(0,0.5, color = 'green')
plt.scatter(1,-0.5, color = 'green')
plt.scatter(2,0.5, color = 'green')
plt.scatter(3,-0.5, color = 'green')
plt.scatter(-1,-0.5, color = 'green')
plt.scatter(-2,0.5, color = 'green')

#x = np.arange(-9, 9, 0.01) #Main interval for point 4
x = np.arange(-2, 4, 0.01) #Main interval for point 5

#Build the generalized Fourier series based on the given ortonormal set
def generalized2():
    for i in a:
        y = np.cos(np.pi*x)/2

        for j in range(1,i+1):
            y = y + (4/np.pi)*(j/(4*j**2-1))*np.sin(2*j*np.pi*x)
        plt.plot(x,y)

def generalized():
    for i in a:
        y = 1/2

        for j in range(1,i+1):
            y = y + (2/np.pi)*(1/(2*j-1))*np.sin((2*j-1)*np.pi*x/3)
        plt.plot(x,y)

def cosine():
    for i in a:
        y = np.pi**4/5

        for j in range(1, i+1):
            y = y + 8*( (-1)**j * (j**2 * np.pi**2 - 6)/(j**4) * np.cos(j*x) )

        plt.plot(x,y)

#generalized2()
#generalized()
cosine()

plt.title('F(x) FOURIER GENERALIZED SERIES') #Plot and display the data
plt.grid(color = 'black', linestyle = '-')
#plt.legend(lab)
plt.show()
