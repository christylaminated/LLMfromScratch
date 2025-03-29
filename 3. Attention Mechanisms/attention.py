import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x.sum(dim=0))

'''
calculate the attention scores/how much each word relates to every other word
1. start at the first vector, create a loop to calculate the attention score between first vector and every other vector
create a loop1 (query) to loop through all vectors:
    create a loop2 (key) to loop through all vectors:
        - calculate the dot product between loop1 vectors (query) and all of loop2 vectors(key)


'''
attention_scores = torch.empty(6, 6)


for i, x_i in enumerate(inputs):
    for y, y_i in enumerate(inputs):
        attention_scores[i,y] = torch.dot(x_i, y_i)

normalized_attention = torch.softmax(attention_scores, dim=-1)
#dim=-1 means last dimension, so each row
print(attention_scores)
print(normalized_attention)

#normalize the attention scores

#you can also use matrix multiplication (@)
attention_scores2 = torch.empty(6,6)
attention_scores2 = inputs @ inputs.T
print(attention_scores2) #with matrix multiplcation

#we now have the attention scores and need to calculate the context vectors
#context vectors: attention weight * input vector
print("shapes: ", normalized_attention.shape, inputs.shape)
context_vector = normalized_attention @ inputs
print(context_vector)






