import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
#calculate all context vectors


x_2 = inputs[1] #[0.55, 0.87, 0.66] -> vector for word jounrey
d_in = inputs.shape[1] #shape of inputs, there are 3 values per word
d_out = 2 #output embedding size for word

#initialize the weight matricies for query, key, and value
W_Query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False) #random matrix with shape (3, 2)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print("W_query shape: ", W_Query.shape)
print("input shape: ", inputs.shape)
print("x_2 shape: ", x_2.shape)

query_2 = x_2 @ W_Query # [0.55, 0.87, 0.66] with shape (1, 3) @ (3, 2) = (1, 2)
#key_2 = x_2 @ W_key
#value_2 = x_2 @ W_value
print("query_2: ", query_2) #shape is [x1, x2]

#the output for the query results in a two dimenstional vector since we set the number of columns of thecorresponding weight matrix, via d_out to 2
keys = inputs @ W_key #inputs (shape = (5, 3)) @ (W_key shape = (3, 2)) so (5, 2) is the final shape
values = inputs @ W_value #(5, 2) is the final shape
print("keys.shape is: ", keys.shape)
print("values.shape: ", values.shape)

#compute the attention scores

keys_2 = keys[1] #key value for the second word
attention_score_22 = query_2.dot(keys_2)  
print(attention_score_22)

#generalize this computation to all attention scores via matrix multiplication
attention_score_2 = query_2 @ keys.T #query_2 = [x1, x2] (1, 2), keys.T = (2, 5)
'''
.T means transpose it, example:
keys =[[1,2],
      [3,4],
      [5, 6]]
keys.T = [[1, 3, 5],
          [2, 4, 6]]

'''
print("all attention scores for the second word: ", attention_score_2) #the second element in the output matches the attention score 22 we calculated previosuly

#now we want to go from the attention scores to the attention weights.
#we compute the attention weights by scaling the attention scores and using the softmax function





