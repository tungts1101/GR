import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os 

file_list = os.listdir('dems_data')
file_list = sorted(file_list)
print(file_list)
for file_name in file_list:
    z = np.loadtxt('dems_data/' + file_name)
    x, y = z.shape
    x, y = np.meshgrid(range(y), range(x))
    print(file_name + ': ' + str(z.shape))
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(x, y, z, alpha=0.2)
    ax = plt.gca()
    # ax.hold(True)
    # ax.scatter(point2[0], point2[1], point2[2], color='green')
    plt.show()


x = [1    2    3    4   15]
y = [6.0000   7.0000   8.0000   9.0000   1.5000]
z = [15   25    7    9    1]
plot3(x,y,z,'o','MarkerSize',10, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', [1 .6 .6])