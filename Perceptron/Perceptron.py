import numpy as np

class Perceptron(object):
    def __init__(self,no_of_inputs,threshold=100,lr=100):
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
        print("\nWeights Before  training " + str(self.w) )
        for _ in range(self.threshold):
            for inputs,label in zip(training_inputs,labels):
                prediction = self.predict(inputs)
                self.w[1:] += self.lr*(label-prediction)*inputs                
                self.w[0] += self.lr*(label-prediction)
                
            if(True):
                    print("\nWeights After " +str(_)+" iteration" + str(self.w))
        print("\n Weights after  training " + str(self.w))
