from importlib.metadata import version

print("torch version:", version("torch"))

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
attn_scores = torch.empty(inputs.shape[0]) #create a matrix that matches to the shape of the 
for i, x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i, query)
print(attn_scores)



