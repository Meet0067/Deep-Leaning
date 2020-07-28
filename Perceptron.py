import numpy as np

class Perceptron(object):
    def __init__(self,no_of_inputs,threshold=100,lr=0.01):
        self.threshold = threshold
        self.lr = lr
        self.w = np.zeros(no_of_inputs + 1)
        
    def predict(self,inputs):
        summation = np.dot(inputs,self.w[1:]) + self.w[0]
        if summation >0:
            activation = 1
        else:
            activation = 0
        return activation
    
    def train(self,training_inputs,labels):
        print("Weights Before  training " + str(self.w) )
        for _ in range(self.threshold):
            for inputs,label in zip(training_inputs,labels):
                prediction = self.predict(inputs)
                self.w[1:] += self.lr*(label-prediction)*inputs                
                self.w[0] += self.lr*(label-prediction)
                
            if(True):
                    print("Weights After " +str(_)+" iteration" + str(self.w))
        print("Weights after  training " + str(self.w))

training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,0]))

labels = np.array([1,0,0,0])
perceptron = Perceptron(no_of_inputs=2,threshold=5)
perceptron.train(training_inputs,labels)

inputss = np.array([1,1])
print("For input "+str(inputss)+" Perceptron Predicted ==>  " + str(perceptron.predict(inputss)))

'''

Weights Before  training [0. 0. 0.]
Weights After 0 iteration[-0.01  0.    0.  ]
Weights After 1 iteration[-0.01  0.01  0.  ]
Weights After 2 iteration[-0.02  0.01  0.  ]
Weights After 3 iteration[-0.02  0.01  0.01]
Weights After 4 iteration[-0.02  0.02  0.01]
Weights after  training [-0.02  0.02  0.01]
For input [1 1] Perceptron Predicted ==>  1


'''

'''
        Observations :
            For this 'AND' Problem if you set learning rate too high or too law ,perceptron will learn till max 5 iterations.
         
'''
