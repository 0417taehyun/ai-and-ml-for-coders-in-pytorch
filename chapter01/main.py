import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


model = nn.Sequential(nn.Linear(in_features=1, out_features=1))

criterion = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.01)

xs = torch.tensor(data=[[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]],
                  dtype=torch.float32)
ys = torch.tensor(data=[[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]],
                  dtype=torch.float32)

for _ in range(500):
    optimizer.zero_grad()

    outputs = model(xs)
    loss = criterion(outputs, ys)
    loss.backward()

    optimizer.step()

with torch.no_grad():
    print(model(torch.tensor([10.0], dtype=torch.float32)))

layer = model[0]

weights = layer.weight.data.numpy()
print("Weights: ", weights)

bias = layer.bias.data.numpy()
print("Bias: ", bias)
