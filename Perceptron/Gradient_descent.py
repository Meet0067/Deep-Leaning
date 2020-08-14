import numpy as np

X = [0.5,2.5]
Y = [0.2,0.9]

def f(w,b,x):
    return 1.0/(1.0 + np.exp(-(w*x + b)))
def error(w,b):
    err = 0.0
    
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5*(fx-y)**2
    
    return err 

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*(fx)*(1-fx)


def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return x*(fx-y)*(fx)*(1-fx)

def do_gradient_descent():
    w,b,eta,max_epoch = -2,-1,1.0,100
    
    for i in range(max_epoch):
        dw = 0
        db = 0
        for x,y in zip(X,Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
            
        w = w - eta*dw        
        b = b - eta*db
        cost = error(w,b)
        print("weights after "+str(i)+" epoch "+str(w))
        print("biase after "+str(i)+" epoch "+str(b))
        print("cost is " + str(cost)) 
        print(" ")
do_gradient_descent()