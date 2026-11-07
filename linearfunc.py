import numpy as np
import matplotlib.pyplot as plt

#define a linear function with just one input, x
def linear_function_1D(x, m, b) :
   y = m * x + b
   return y
   
# Plot the 1D linear function
# define an array of x values from 0 to 10 with a step of 0.01

x = np.arange(0, 10, 0.01)
m = -2  # slope
b = 10  # y-intercept
y = linear_function_1D(x, m, b)

#plot this function using matplotlib
fig, ax = plt.subplots()
ax.plot (x, y, label=f'y = {m}x + {b}')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_title('1D Linear Function')
ax.legend()
ax.grid(True)  
plt.show()
