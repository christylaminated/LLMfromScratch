import torch

inputs = torch.tensor()
#calculate all context vectors


x_2 = inputs[1]
d_in = inputs.shape[1] #shape of inputs
d_out = 2 #output embedding size

#initialize the weight matricies for query, key, and value
W_Query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)



query_2 = x_2 @ W_Query


