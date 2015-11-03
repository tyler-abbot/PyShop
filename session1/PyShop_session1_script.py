"""
Origin: A simple example.
Filename: example_scatter.py
Author: Tyler Abbot
Last modified: 15 September, 2015
"""
import matplotlib.pyplot as plt
import random

x, y = [], []
for i in range(0,50):
   #This is a comment
   x.append(random.random())
   y.append(random.random())

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('A Bivariate Uniform Random Variable')
plt.show()