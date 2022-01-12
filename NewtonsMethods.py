from queue import Queue

from sympy import sin, cos, Matrix, symbols, lambdify, simplify, expand, factorial, combsimp, Abs, refine
import math
import numpy
import multiprocessing

def create_file_with_SBP(n):
    f = open("SBP.txt", 'w')
    strings=[]
    strings.append("((3-2*x1)*x1-2*x2+1)**2\n")
    for i in range(1,n-1):
        strings.append("((3-2*x"+str(i+1)+")*x"+str(i+1)+"-x"+str(i)+"-2*x"+str(i+2)+"+1)**2\n")
    strings.append("((3-2*x"+str(n)+")*x"+str(n)+"-x"+str(n-1)+"+1)**2\n")

    string=""
    for i in range(n):
        string+="x"+str(i+1)+" "
    strings.append(string[:-1]+"\n")


    string=""
    for i in range(n):
        string+="-1 "
    strings.append(string[:-1])

    for string in strings:
        f.write(string)
    f.close()
def create_file_with_JS(n):
    f = open("JS.txt", 'w')
    strings=[]
    strings.append("((3-2*x1)*x1-2*x2+1)**2\n")
    for i in range(1,n-1):
        strings.append("((3-2*x"+str(i+1)+")*x"+str(i+1)+"-x"+str(i)+"-2*x"+str(i+2)+"+1)**2\n")
    strings.append("((3-2*x"+str(n)+")*x"+str(n)+"-x"+str(n-1)+"+1)**2\n")

    string=""
    for i in range(n):
        string+="x"+str(i+1)+" "
    strings.append(string[:-1]+"\n")


    string=""
    for i in range(n):
        string+="-1 "
    strings.append(string[:-1])

    for string in strings:
        f.write(string)
    f.close()
def read_file(pathfile):
    args = []
    with open(pathfile, "r") as f:
        for line in f:
            args+=[line.replace("\n","")]
    return args
def create_data(args):

    F_exp=Matrix(args[:-2])
    Y = Matrix(args[-2].split(" "))
    dF_exp = F_exp.jacobian(Y)
    symbs=args[-2]
    x0=args[-1]
    return F_exp, dF_exp,symbs,x0
def calc_alt_dfij(j,x,xk,xk1,Fi):
    ready_data = ready_data_for_df(j,x, xk, xk1)
    Fi1=Fi.subs(ready_data)

    ready_data2 = ready_data_for_df(j-1, x, xk, xk1)
    Fi2=Fi.subs(ready_data2)

    return (Fi1-Fi2)/(xk[j]-xk1[j])
def ready_data_for_df(j, x, xk, xk1):
    ready_data = []
    for i in range(j+1):
        ready_data.append((x[i], xk[i]))
    for i in range(j+1, len(x)):
        ready_data.append((x[i], xk1[i]))
    return ready_data
def create_alternative_df(x, xk, xk1,xk2, F_exp):
    x = x.split()
    matrix=[]
    for Fi in F_exp:
        line=[]
        for j in range(len(x)):
            elem=calc_alt_dfij(j,x,xk,xk1,Fi)+calc_alt_dfij(j,x,xk2,xk,Fi)-calc_alt_dfij(j,x,xk2,xk1,Fi)
            line.append(elem)
        matrix.append(line)
    return matrix
def create_alternative_df2(x, xk, xk1, F_exp):
    x = x.split()
    matrix=[]
    for Fi in F_exp:
        line=[]
        for j in range(len(x)):
            elem=calc_alt_dfij(j,x,xk1,xk,Fi)
            line.append(elem)
        matrix.append(line)
    return matrix
def show_info(info):
    print("\nInterations:\n")
    for elem in info:
        print(elem[0])
        print(elem[1])
        print(elem[2])
        print()
def newton_method(F, J, x0, eps):
    xk=x0
    x=[0]*len(xk)
    info=[]
    iters=0

    normx = math.fabs(numpy.linalg.norm([e1 - e2 for e1, e2 in zip(xk, x)], numpy.inf))
    while (normx > eps):
        x = xk
        F_value_minus=[-1*e for l in F(x) for e in l]
        J_value = J(x)
        b=numpy.linalg.lstsq(J_value, F_value_minus, rcond=None)[0]

        xk=[e1 + e2 for e1, e2 in zip(x, b)]
        iters+=1
        normx = math.fabs(numpy.linalg.norm([e1 - e2 for e1, e2 in zip(xk, x)], numpy.inf))
        info.append([iters, xk, normx])
    return xk, info
def potra_method(symb,F,F_exp, x0, eps):
    xk = x0
    x = [0] * len(xk)
    xk1=[e+10**-4 for e in xk]
    xk2=[e+2*10**-4 for e in xk]
    info=[]
    iters=0
    normx = math.fabs(numpy.linalg.norm([e1 - e2 for e1, e2 in zip(xk, x)], numpy.inf))
    while (normx >= eps):
        x = xk
        F_value_minus = [-1 * e for l in F(x) for e in l]
        J_value = numpy.matrix(create_alternative_df(symb, xk, xk1,xk2, F_exp), dtype='float')
        b = numpy.linalg.lstsq(J_value, F_value_minus, rcond=None)[0]

        xk2=xk1
        xk1=xk
        xk = [e1 + e2 for e1, e2 in zip(x, b)]
        iters += 1
        normx = math.fabs(numpy.linalg.norm([e1 - e2 for e1, e2 in zip(xk, x)], numpy.inf))
        info.append([iters, xk, normx])

    return xk, info
def potra_method2(symb,F,F_exp, x0, eps):
    xk = x0
    x = [0] * len(xk)
    xk1=[e+10**-4 for e in xk]
    info=[]
    iters=0
    normx = math.fabs(numpy.linalg.norm([e1 - e2 for e1, e2 in zip(xk, x)], numpy.inf))
    while (normx > eps):
        x = xk
        F_value_minus = [-1 * e for l in F(x) for e in l]
        J_value = numpy.matrix(create_alternative_df2(symb, xk, xk1, F_exp), dtype='float')
        b = numpy.linalg.solve(J_value, F_value_minus)

        xk1=xk
        xk = [e1 + e2 for e1, e2 in zip(x, b)]
        iters += 1
        normx = math.fabs(numpy.linalg.norm([e1 - e2 for e1, e2 in zip(xk, x)], numpy.inf))
        info.append([iters, xk, normx])

    return xk, info
def pro_newton_method(F, J, x0, eps):

    xk=numpy.array(x0)
    Ak=numpy.linalg.inv(J(xk))
    x=numpy.array([0]*len(xk))
    info=[]
    iters=0
    I2 = numpy.identity(len(x0))*2
    normx = math.fabs(numpy.linalg.norm(numpy.array([e1 - e2 for e1, e2 in zip(xk, x)]), numpy.inf))
    while (normx > eps):
        x = xk
        Ak_F=Ak.dot(numpy.array([e for l in F(x) for e in l]))
        xk=numpy.array([e1 - e2 for e1, e2 in zip(x, Ak_F)])

        A=Ak
        Ak=A.dot(I2-J(xk).dot(A))
        iters+=1

        normx = math.fabs(numpy.linalg.norm(numpy.array([e1 - e2 for e1, e2 in zip(xk, x)])))
        info.append([iters, xk, normx])
    return xk, info
def pro_potra_method(symb,F,F_exp, x0, eps):
    xk=numpy.array(x0)
    x = numpy.array([0] * len(xk))
    xk1=numpy.array([e+10**-4 for e in xk])
    xk2=numpy.array([e+2*10**-4 for e in xk])
    Ak = numpy.linalg.inv(numpy.array(create_alternative_df(symb, xk, xk1,xk2, F_exp), dtype='float'))
    I2 = numpy.identity(len(x0)) * 2

    info=[]
    iters=0

    normx = math.fabs(numpy.linalg.norm(numpy.array([e1 - e2 for e1, e2 in zip(xk, x)]), numpy.inf))
    while (normx > eps):
        x = xk
        Ak_F = Ak.dot(numpy.array([e for l in F(x) for e in l]))
        xk2=xk1
        xk1=xk
        xk = numpy.array([e1 - e2 for e1, e2 in zip(x, Ak_F)])

        A = Ak
        Ak = A.dot(I2 - numpy.array(create_alternative_df(symb, xk, xk1,xk2, F_exp), dtype='float').dot(A))
        iters += 1

        normx = math.fabs(numpy.linalg.norm(numpy.array([e1 - e2 for e1, e2 in zip(xk, x)])))
        info.append([iters, xk, normx])

    return xk, info
def xk_for_parallel(xk,Ak, F, q):
    Ak_F = Ak.dot(numpy.array([e for l in F(xk) for e in l]))
    q.put(numpy.array([e1 - e2 for e1, e2 in zip(xk, Ak_F)]))
def Ak_for_parallel_newton(xk, I2, Ak, J, q):
    q.put(Ak.dot(I2 - J(xk).dot(Ak)))
def parallel_pro_newton_method(F, J, x0, eps):

    xk=numpy.array(x0)
    Ak=numpy.linalg.inv(J(xk))
    x=numpy.array([0]*len(xk))
    info=[]
    iters=0
    I2 = numpy.identity(len(x0))*2
    normx = math.fabs(numpy.linalg.norm(numpy.array([e1 - e2 for e1, e2 in zip(xk, x)]), numpy.inf))

    processes=[]
    q1 = Queue()
    q2 = Queue()
    while (normx > eps):
        x=xk
        q1=Queue()
        q2=Queue()
        process1 = multiprocessing.Process(xk_for_parallel( xk,Ak, F, q1))
        process2 = multiprocessing.Process(Ak_for_parallel_newton(xk, I2, Ak, J, q2))
        process1.start()
        process2.start()
        processes.append(process1)
        processes.append(process2)
        xk=q1.get()
        Ak=q2.get()

        iters+=1
        normx = math.fabs(numpy.linalg.norm(numpy.array([e1 - e2 for e1, e2 in zip(xk, x)])))
        info.append([iters, xk, normx])
    for p in processes:
        p.kill()
    return xk, info
def Ak_for_parallel_potra(xk,xk1, xk2, I2, Ak, symb, F_exp, q):
    q.put(Ak.dot(I2 - numpy.array(create_alternative_df(symb, xk, xk1,xk2, F_exp), dtype='float').dot(Ak)))
def parallel_pro_potra_method(symb,F,F_exp, x0, eps):
    xk=numpy.array(x0)
    x = numpy.array([0] * len(xk))
    xk1=numpy.array([e+10**-4 for e in xk])
    xk2=numpy.array([e+2*10**-4 for e in xk])
    Ak = numpy.linalg.inv(numpy.array(create_alternative_df(symb, xk, xk1,xk2, F_exp), dtype='float'))
    I2 = numpy.identity(len(x0)) * 2

    info=[]
    iters=0
    normx = math.fabs(numpy.linalg.norm(numpy.array([e1 - e2 for e1, e2 in zip(xk, x)]), numpy.inf))
    processes=[]
    q1 = Queue()
    q2 = Queue()
    while (normx > eps):
        x=xk

        process1 = multiprocessing.Process(xk_for_parallel(xk, Ak, F, q1))
        process2 = multiprocessing.Process(Ak_for_parallel_potra(xk,xk1, xk2, I2, Ak, symb, F_exp, q2))
        process1.start()
        process2.start()
        processes.append(process1)
        processes.append(process2)
        xk2=xk1
        xk1=xk
        xk = q1.get()
        Ak = q2.get()


        iters += 1

        normx = math.fabs(numpy.linalg.norm(numpy.array([e1 - e2 for e1, e2 in zip(xk, x)])))
        info.append([iters, xk, normx])
    for p in processes:
        p.kill()
    return xk, info



def calc_alt_dfij3(j,x,xk,xk1,Fi):
    ready_data = ready_data_for_df(j,x, xk, xk1)
    Fi1=simplify(Fi.subs(ready_data))
    ready_data2 = ready_data_for_df(j-1, x, xk, xk1)
    Fi2=simplify(Fi.subs(ready_data2))

    return (Fi1-Fi2)/(xk[j]-xk1[j])
def create_alternative_df3(x, xk, xk1,xk2, F_exp):
    x = x.split()
    matrix=[]
    for Fi in F_exp:
        line=[]
        for j in range(len(x)):
            elem=calc_alt_dfij3(j,x,xk,xk1,Fi)+calc_alt_dfij3(j,x,xk2,xk,Fi)-calc_alt_dfij3(j,x,xk2,xk1,Fi)
            line.append(elem)
        matrix.append(line)
    return matrix

def newton_potra_method(symb,F, J, G,G_exp, x0, eps):
    xk = numpy.array(x0)
    x = numpy.array([0] * len(xk))
    xk1=numpy.array([e+10**-4 for e in xk])
    xk2=numpy.array([e+2*10**-4 for e in xk])
    info=[]
    iters=0
    normx = math.fabs(numpy.linalg.norm(xk-x, numpy.inf))
    while (normx >= eps):
        x = xk
        F_value=numpy.array([e for l in F(x) for e in l])
        G_value=numpy.array([e for l in G(x) for e in l])
        H_value = F_value + G_value

        J_value=numpy.matrix(J(x), dtype="float")
        G_alternative_value = numpy.matrix(create_alternative_df3(symb, xk, xk1,xk2, G_exp), dtype="float")
        inverse_matrix=numpy.linalg.inv(J_value+G_alternative_value)
        xk2=xk1
        xk1=xk
        xk = (numpy.array(xk1) - numpy.array(inverse_matrix.dot(H_value)))
        xk = numpy.array([e for l in xk for e in l])
        iters += 1
        normx = math.fabs(numpy.linalg.norm(xk-x, numpy.inf))
        info.append([iters, xk, normx])

    return xk, info
def potra_method3(symb,F, G,G_exp, F_exp, x0, eps):
    xk = numpy.array(x0)
    x = numpy.array([0] * len(xk))
    xk1=numpy.array([e+10**-4 for e in xk])
    xk2=numpy.array([e+2*10**-4 for e in xk])
    info=[]
    iters=0
    normx = math.fabs(numpy.linalg.norm(xk-x, numpy.inf))
    while (normx >= eps):
        x = xk
        F_value=numpy.array([e for l in F(x) for e in l])
        G_value=numpy.array([e for l in G(x) for e in l])
        H_value = F_value + G_value

        F_alternative_value = numpy.matrix(create_alternative_df3(symb, xk, xk1, xk2, F_exp), dtype="float")
        G_alternative_value = numpy.matrix(create_alternative_df3(symb, xk, xk1,xk2, G_exp), dtype="float")

        inverse_matrix=numpy.linalg.inv(F_alternative_value+G_alternative_value)
        xk2=xk1
        xk1=xk
        xk = (numpy.array(xk1) - numpy.array(inverse_matrix.dot(H_value)))
        xk = numpy.array([e for l in xk for e in l])
        iters += 1
        normx = math.fabs(numpy.linalg.norm(xk-x, numpy.inf))
        info.append([iters, xk, normx])

    return xk, info

def gauss_newton_method(F, J, x0, eps):
    xk = numpy.array(x0)
    x = numpy.array([0] * len(xk))
    info=[]
    iters=0

    Ak = numpy.matrix(J(x), dtype="float")
    Akt = numpy.matrix.transpose(Ak)
    Fv = numpy.array([e for l in F(xk) for e in l])
    normx = math.fabs(numpy.linalg.norm(numpy.array(Akt.dot(Fv)).reshape(-1)))
    print(normx)
    #normx = math.fabs(numpy.linalg.norm(xk-x, numpy.inf))
    while (normx > eps):
        x = xk
        Fv=numpy.array([-1*e for l in F(x) for e in l])

        Ak=numpy.matrix(J(x), dtype="float")
        Akt=numpy.matrix.transpose(Ak)

        AktAk=Akt*Ak


        AkF=numpy.array(Akt.dot(Fv)).reshape(-1)

        b=numpy.linalg.lstsq(AktAk, AkF, rcond=None)[0]
        #b = numpy.linalg.solve(AktAk, AkF)

        xk=x+b

        iters+=1
        Fv = numpy.array([e for l in F(xk) for e in l])
        normx = math.fabs(numpy.linalg.norm(AkF))
        # normx = math.fabs(numpy.linalg.norm(xk - x, numpy.inf))
        norm = math.fabs(numpy.linalg.norm(Fv))
        info.append([iters, xk, norm])
    return xk, info
def potra_method4(symb,F, F_exp, x0, eps):
    xk = numpy.array(x0)
    x = numpy.array([0] * len(xk))
    xk1=numpy.array([e+10**-4 for e in xk])
    xk2=numpy.array([e+2*10**-4 for e in xk])
    info=[]
    iters=0

    Ak = numpy.matrix(create_alternative_df3(symb, xk, xk1, xk2, F_exp), dtype="float")
    Akt = numpy.matrix.transpose(Ak)
    Fv = numpy.array([e for l in F(x) for e in l])

    normx = math.fabs(numpy.linalg.norm(numpy.array(Akt.dot(Fv)).reshape(-1)))

    # normx = math.fabs(numpy.linalg.norm(xk-x, numpy.inf))

    while (normx > eps):
        x = xk
        Fv = numpy.array([-1 * e for l in F(x) for e in l])

        Ak = numpy.matrix(create_alternative_df3(symb, xk, xk1, xk2, F_exp), dtype="float")
        Akt = numpy.matrix.transpose(Ak)

        AktAk = Akt * Ak
        AktF = numpy.array(Akt.dot(Fv)).reshape(-1)
        b = numpy.linalg.lstsq(AktAk, AktF, rcond=None)[0]
        #b = numpy.linalg.solve(AktAk, AktF)
        xk = x + b

        iters += 1
        # normx = math.fabs(numpy.linalg.norm(xk - x, numpy.inf))
        Fv = numpy.array([e for l in F(xk) for e in l])
        normx = math.fabs(numpy.linalg.norm(numpy.array(AktF)))
        norm = math.fabs(numpy.linalg.norm(Fv))

        info.append([iters, xk, norm])

    return xk, info