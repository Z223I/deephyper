## forward(x)
## input_0 = x
##input_0 = self.layer_norm(input_0)
##dense = self.dense(input_0)
##dense = self.dropout_1(dense)
##activation = self.activation(dense)

##dense_1 = self.dense_1(input_0)
##dense_1 = self.dropout_2(dense_1)
##add = self.add(activation, dense_1)

## Can use ReLU(inplace=False)
##activation_1 = self.activation_1(add)

##dense_2 = self.dense_2(activation_1)
##dense_2 = self.dropout_3(dense_2)
##activation_2 = self.activation_2(dense_2)

##dense_3 = self.dense_3(activation_1)
##dense_3 = self.dropout_4(dense_3)
##add_1 = self.add_1(activation_2, dense_3)
## Can use ReLU(inplace=False)
##activation_3 = self.activation_3(add_1)

##dense_4 = self.dense_4(activation_3)
##dense_4 = self.dropout_5(dense_4)
##activation_4 = self.activation_4(dense_4)

##dense_5 = self.dense_5(input_0)
##dense_5 = self.dropout_6(dense_5)

##dense_6 = self.dense_6(activation_3)
##dense_6 = self.dropout_7(dense_6)
##add_2 = self.add_2(activation_4, dense_5, activation_1, dense_6)
### Can use ReLU(inplace=False)
##activation_5 = self.activation_5(add_2)

##dense_7 = self.dense_7(activation_5)







#"input_layers": [["input_0", 0, 0]],
self.layer_norm = nn.LayerNorm(input_shape).to(device, dtype=dtype)

#if batchSamples >= 16:
in_features = numInputs
out_features = 80
self.dense = nn.Linear(in_features, out_features).to(device, dtype=dtype)
self.dropout_1 = nn.Dropout2d(dropout1).to(device, dtype=dtype)

self.activation = torch.nn.Sigmoid().to(device, dtype=dtype)


#if batchSamples >= 12:
in_features = out_features
out_features = 80
self.dense_1 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
input_0 = activation
self.dropout_2 = nn.Dropout2d(dropout2).to(device, dtype=dtype)


# Add
out_features = 1
self.add = torch.nn.Sum().to(device, dtype=dtype)

out_features = 1
self.activation_1 = torch.nn.ReLU().to(device, dtype=dtype)

#if batchSamples >= 10:
in_features = out_features
out_features = 96
self.dense_2 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
self.dropout_3 = nn.Dropout2d(dropout2).to(device, dtype=dtype)

out_features = 1
self.activation_2 = torch.nn.Tanh().to(device, dtype=dtype)

in_features = out_features
out_features = 96
self.dense_3 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
self.dropout_4 = nn.Dropout2d(dropout3).to(device, dtype=dtype)

out_features = 1
self.add_1 = torch.nn.Sum().to(device, dtype=dtype)

out_features = 1
self.activation_3 = torch.nn.ReLU().to(device, dtype=dtype)

in_features = out_features
out_features = 80
self.dense_4 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
self.dropout_5 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

out_features = 1
self.activation_4 = torch.nn.Sigmoid().to(device, dtype=dtype)

in_features = out_features
out_features = 80
self.dense_5 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
self.dropout_6 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

in_features = out_features
out_features = 80
self.dense_6 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
self.dropout_7 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

out_features = 1
self.add_2 = torch.nn.Sum().to(device, dtype=dtype)

out_features = 1
self.activation_5 = torch.nn.ReLU().to(device, dtype=dtype)


in_features = out_features
out_features = 2 # This is 2 instead of 1 due to the PyTorch softmax.
self.dense_7 = nn.Linear(in_features, out_features).to(device, dtype=dtype)

#"output_layers": [["dense_7", 0, 0]]}