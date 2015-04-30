# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:30:01 2015

@author: skywalkerhunter
"""

"""
Simple demo with multiple subplots.
"""
import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

x_num = 3
y_num = 1
X_size = 2.0
Y_size = 8.0

plt.subplots(x_num, y_num, sharey = True, figsize=(y_num * Y_size, x_num * X_size) )

plt.subplot(x_num, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(x_num, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.subplot(x_num, 1, 3)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()


