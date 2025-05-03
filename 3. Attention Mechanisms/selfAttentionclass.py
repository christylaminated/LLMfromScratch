import torch
import torch.nn as nn
#our models should subclass the class nn.Module
'''
class subClassName(baseClass):
    def __init__(self):
        super().__init__()

Java equivalent:
    public class SubClass extends BaseClass {
        public SubClass() {
            super(); // call parent class constructor
        }
}

'''

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        #d_in and d_out are passed in as arguments when we create an instance of this class
        super().__init__() #inherit all functions of nn.Module, nn.Linear, nn.2dConv
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) #create matrix with shape d_in and d_out
        self.W_key = nn.Parameter(torch.rand(d_in, d_out)) 
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        #pass in token embeddings to get context aware embeddings and self (so were able to access self.W_query and self.W_key)
        #X is the input represented as token embedding of the sentence. each row represents one word and each column in each row represents the features that make up that word in the row
        # x has shape (sequence_length, d_in) and W_key has shape (d_in, d_out)
        '''
        x represents all of the words in the sentence
        d_in is how many columns there are per word 
        '''
        keys = x @ self.W_key #keys = each word of the input token embedding projected by the key's random weight matrix
        queries = x @ self.W_query #each word of the input token embedding projected by the query's random weight matrix
        values = x @ self.W_value #each word of the input token embedding projected by the value's random weight matrix
        # keys, queries, and values -> (sequence_length, d_in) @ (d_in, d_out) = (sequence_length, d_out)
    
        attn_scores = queries @ keys.T # attention score matrix -> often denoted as omega, keys.T = (d_out, sequence_length)
        # attn_scores = (sequence_length, d_out) @ (d_out, sequence_length) = (sequence_length, sequence_length)
        # attn_scores = dot product of queries and keys to see how simular they are, how much attention that token gives to other tokens
        # attn_scores[i][j] = how much token i pays attention to token j, but raw scores
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim = -1
        )
        #sqrt(d_out) -> divide the attention_scores by the square root of how many features there are per word
        # if d_out get too large, attentionscores get too large (after dot product) or too negative
        #so if attention score is too positive or too negative, it will turn very close to 0 after softmax
        # we want the number in softmax(here) to be as close to 0 as possible
        # everything in each row of attn_weights will add up to 1
        context_vector = attn_weights @ values 
        #Each row is how much each word (at that index) should pay attention to every other word, including itself — 
        # and every column represents how much attention is paid to that word, depending on which row you're at.
        return context_vector
    
    


#use the class
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(123)
d_in = inputs.shape[1] # 3
d_out = 2 #input and output dimensions are usually the same but in this exmaple, different to better visualize
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
'''
output = sa_v1.__call__(inputs)   # nn.Module's __call__
→ calls self.forward(inputs)      # your implementation bc SelfAttention_v1 is a subclass of nn.Module
'''

'''
tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)
it the context vector, each row representing how much attention to pay to each word
'''

