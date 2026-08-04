import matplotlib.pyplot as plt
import numpy as np

x=[1,2,3]
y=[9,8,7]

plt.plot(x,y)
for a,b in zip(x, y): 
    plt.text(a, b, str(b))


plt.annotate(str(np.mean(x)), xy=(2,8), xytext=(3,8.5),
            )
plt.show()
