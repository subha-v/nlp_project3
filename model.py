import torch
import torch.nn as nn

class RNN_GEN(nn.Module):
    def __init__(self, num_countries, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.input_to_hidden = nn.Linear(num_countries+input_dim+hidden_dim, hidden_dim) # this takes our input, an input layer, the hidden dimension, and 
        self.input_to_output = nn.Linear(num_countries + input_dim + hidden_dim, output_dim)
        self.output_to_output = nn.Linear(hidden_dim+output_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.hidden_dim = hidden_dim
#our hidden state exists outside of our model 
#we take an input letter and then output the next letter
#we let the computer generate some hidden layer
    def forward(self, inp, country_tensor, hidden):
        combo = torch.cat([country_tensor, inp, hidden], dim=1) #concatenating these three arrays  , its dim=1 because without a dimension it assumes it is 0
        #we are ALSO INCORPORATING THE COUNTRY TENSOR BECAUSE THE COUNTRY IS IMPORTANT TO GENERATE A NAME,
        #generatoing greek and korean names are hte same
        hidden = self.input_to_hidden(combo) # we treat combo as the input of a function
        out = self.input_to_output(combo)
        out_combo = torch.cat([hidden, out], dim=1) # [1,hidden dim + output dim]
        out = self.output_to_output(out_combo) #[1, output_di]
        out = self.log_softmax(out) # we put this in the nonlinear 
        #THEE TWO TOP LINES HAVE extra flexibility as more parameters to move
        return out, hidden
        #now our output is a 1x28 array with probabilities for each letter

    def init_hidden(self): #returns a tensor of all 0s with 
        return torch.zeros((1,self.hidden_dim))



