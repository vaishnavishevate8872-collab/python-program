import matplotlib.pyplot as plt
import numpy as np
t=np.linspace(0,2*np.pi,200)
x=16*np.sin(t)**3
y=13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
plt.plot(x,y,'r',linewidth=3)
plt.title("I Love Python",fontsize=16)
plt.axis('equal')
plt.show()