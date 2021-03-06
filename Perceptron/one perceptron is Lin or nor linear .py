import numpy as np
from perceptron import Perceptron

training_inputs=[]
training_inputs.append(np.array([0,0,0,0]))
training_inputs.append(np.array([0,0,0,1]))
training_inputs.append(np.array([0,0,1,1]))
training_inputs.append(np.array([0,1,0,0]))
training_inputs.append(np.array([0,1,0,1]))
training_inputs.append(np.array([0,1,1,0]))
training_inputs.append(np.array([0,1,1,1]))
training_inputs.append(np.array([1,0,0,0]))
training_inputs.append(np.array([1,0,0,1]))
training_inputs.append(np.array([1,0,1,0]))
training_inputs.append(np.array([1,0,1,1]))
training_inputs.append(np.array([1,1,0,0]))
training_inputs.append(np.array([1,1,0,1]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([1,1,1,1]))

labels=np.array([0,1,1,0,0,0,0,1,0,0,1,0,1,0,1,0])
Perceptron1 = Perceptron(4)
Perceptron1.train(training_inputs,labels)

inputs=np.array([0,0,0,0])
print(Perceptron1.predict(inputs))

inputs=np.array([1,1,0,0])
print(Perceptron1.predict(inputs))

inputs=np.array([1,1,1,0])
print(Perceptron1.predict(inputs))

inputs=np.array([1,1,1,1])
print(Perceptron1.predict(inputs))