import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


col = ["1","2","3"]
#col = ["1","2","3","4","5","6"]
#col = ["1","2","3","4","5","6","7","8","9"]
#col = ["1","2","3","4","5","6","7","8","9","10","11","12"]
X = pd.read_csv("./train_data3.csv",sep=",",names = col)
Y = pd.read_csv("./label_data3.csv",sep=",",names = ["label"])


noise_var = 0
num_datapoints = len(X)
test_size = 0.2
num_train = int((1-test_size) * num_datapoints)
num_train = num_train-num_train%100


# Network params
input_size = 3 #length of input vector x
batch_block_size = 100
# If `per_element` is True, then LSTM reads in one timestep at a time.
per_element = True
op_method = 1
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
h1 = 32    #hidden dimension
output_dim = 1
num_layers = 2
learning_rate = 1e-2     
num_epochs = 1000     

#normalization
X = X/50
Y = Y/50



X_train = X.loc[0:num_train-1,:]
y_train = Y.loc[0:num_train-1,:]

X_test = X.loc[num_datapoints-num_train:num_datapoints-1,:]
y_test = Y.loc[num_datapoints-num_train:num_datapoints-1,:]

X_train = torch.tensor(X_train.values,dtype=torch.float)
X_test = torch.tensor(X_test.values,dtype=torch.float)
y_train = torch.tensor(y_train.values,dtype=torch.float).view(-1)
y_test = torch.tensor(y_test.values,dtype=torch.float).view(-1)

X_train = X_train.view([input_size, -1, 1])
X_test = X_test.view([input_size, -1, 1])




class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)


    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)


model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=False)

# choose different optimiser
if op_method == 1:
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
else :
    optimiser = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)



hist = np.zeros(num_epochs)

model.hidden = model.init_hidden()
for t in range(num_epochs):
    
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
        print("mean: ", y_pred.mean())
        print(y_pred)
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()



# training model 
loss_MSE = loss_fn(y_pred, y_train).item()

plt.plot(y_train.detach().numpy(), label="LabelData",linewidth=0.5)
plt.plot(y_pred.detach().numpy(), label="TrainedData",linewidth=0.5)
plt.legend()
plt.show()

start = 100
plt.plot(y_train.detach().numpy()[start:start+100], label="LabelData",linewidth=0.5)
plt.plot(y_pred.detach().numpy()[start:start+100], label="TrainedData",linewidth=0.5)
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()



##########
#testing model
#########
y_test_pred = model(X_test)
loss_MSE = loss_fn(y_test_pred, y_test).item()

print("loss_MSE:", loss_MSE)

plt.plot(y_test.detach().numpy()[1460:2000], label="LabelData",linewidth=0.5)
plt.plot(y_test_pred.detach().numpy()[1460:2000], label="TrainedData",linewidth=0.5)
plt.legend()
plt.show()

end = 350
plt.plot(y_test.detach().numpy()[end-50:end], label="LabelData",linewidth=0.5)
plt.plot(y_test_pred.detach().numpy()[end-50:end], label="TrainedData",linewidth=0.5)
plt.legend()
plt.show()


# -*- coding: utf-8 -*-

