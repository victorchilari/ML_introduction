# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np

X = np.random.normal(loc=1, scale=10, size=(1000, 50))
print(X)
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
m = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = ((X - m)  / std)
print(X_norm)

Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])
print(np.nonzero(Z > 10))

A = np.eye(3)
B = np.eye(3)
print(A)
print(B)

AB = np.vstack((A, B))
print(AB)