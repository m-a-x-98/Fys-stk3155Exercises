The different optimizers available 
ADAM 
ADAgrad 
RMSprop
Momentum 
Basic (just normal stochastic gradient descent)

The different activation functions 
ReLU 
Tanh
Sigmoid
Softmax (with Cross Entropy)

The different loss functions 
MSE 
Cross Entropy

The different layer types 
Dense (fully connected layer)

The different learning rate schedulers 
StepLR 




How to build a network 
```python
model = Network() # define the model

# Define the layers
layer1 = Dense((inshape1, outshape1), activation=ReLU())
layer2 = Dense((outshape1, outshape2), activation=Sigmoid())
...

# Add layers 
model.add(layer1)
model.add(layer2)
...

# Define the loss and the optimizer 
loss = MSE()
optimizer = ADAM(learningRate, beta1, beta2)

# Compile the model 
model.compile(loss, optimizer)

# To check against autograd 
model.checkWithAutograd = False

# Get information about the model
model.summary()
print("="*60)
model.specifics()

# Do one iteration of training 
model.step(x_trian, y_train)

# Evaluate the model
model.evaluate(x_test, y_test)

# Define classes 
classes = [1, 2, 3]
model.defineClasses(classes)

# Use the model
model.predict(data)
model.predictClasses(data) # requires classes to be defined 
```
