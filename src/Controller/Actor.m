function a=Actor(s,k, dim, Param, activation)
[W, B, L] = param2net(Param, dim, activation);
net.weights = W;
net.biases = B;
net.layers = L;
a = NN(net,[s;k],[]);
end