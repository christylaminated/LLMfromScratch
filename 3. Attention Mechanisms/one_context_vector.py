from importlib.metadata import version
import torch

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
attn_scores = torch.empty(inputs.shape[0]) #create a matrix that matches to the shape of the number of words/number of embedding vectors
for i, x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i, query)
print(attn_scores) #unnormalized, but second one is the highest obv
attention_weights_2_tmp = attn_scores / attn_scores.sum() #divides every element in attn_score by total sum
print(attention_weights_2_tmp) #normalized, but in future use softmax function for normalization

#basic softmax implementation
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x.sum(dim=0))

#torch.exp returns a new tensor with expoential of the elements of the input

#now that we have our normalized attention weights of the word "Journey" compared to all other words in the sentence,
#compute context vector: multiplying the embedded input tokens by the coresponding attention weights (how much each word matters)
query = inputs[1] # [0.55, 0.87, 0.66]
context_vec = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec = context_vec + (attention_weights_2_tmp[i] * x_i)
    print(f"context_vec: {context_vec}, attention_weights: {attention_weights_2_tmp[i]}, x_i: {x_i}")

print("context vector:", context_vec)






