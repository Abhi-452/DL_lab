import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X=np.linspace(1,10,100)
Y=2*X+10+np.random.randn(X.shape[0])
print(X)
print(Y)

#defin the architecture of the model
model=Sequential()
model.add(Dense(1,input_dim=1,activation='linear'))

#compile model
model.compile(optimizer='sgd',loss='mse')

#train the model
model.fit(X,Y,epochs=100,verbose=1)

#make predictiion
pred=model.predict(X)

plt.scatter(X,Y,label='original data')
plt.plot(X,pred,label="predicted data ")
plt.show()

