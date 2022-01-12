from abc import ABC, abstractmethod
from sympy import sin, cos, Matrix, symbols, lambdify, simplify, expand, factorial, combsimp
from  NewtonMethods import *
import math
import scipy.sparse.linalg
import scipy.sparse
import scipy
from scipy import optimize
import numpy


if __name__=="__main__":
    path = "1.txt"
    F_exp, J_exp,symb,x0=create_data(read_file(path))
    x0=[float(e) for e in x0.split()]
    F=lambdify([symb.split()],F_exp)
    J=lambdify([symb.split()],J_exp)
    print(F_exp)
    print(J_exp)
    eps=0.001
    print("New Newton - Gauss method\n")
    x, info = newton_method(F,J,x0,eps)
    print("x = ", x)
    print("F(x) = ", F(x))
    show_info(info)


    print("New Potra's method\n")
    x, info = potra_method4(symb,F, F_exp, x0, eps)
    print("x = ", x)
    print("F(x) = ", F(x))
    show_info(info)
