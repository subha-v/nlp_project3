import torch
#think of it like 
# 
# 
# numpy, a lot of it is like pytorch

END_CHAR = "|" #this shows when a name ends

all_letters = 'abcdefghijklmnopqrstuvwxyz ' + END_CHAR
NUM_LETTERS = len(all_letters)

def letter_to_index(letter):
    return all_letters.index(letter)

def letter_to_tensor(letter): #tensor is similar to numpy array
    idx = letter_to_index(letter)
    one_hot=torch.zeros((1,NUM_LETTERS))
    one_hot[0,idx]=1 #the 0 refers to the first dimension and then in the second dimension do index idx
    return one_hot #now we transferred letter into numbers that computer understands

def name_to_tensor(name):
    letter_tensors = []
    for letter in name:
        letter_tensors.append(letter_to_tensor(letter))

    return torch.stack(letter_tensors) #this stacks all the letters in the name into one tensor

    #calculus is the key

#chi -> hi|
def shift_name_right(name):
    return name[1:] + END_CHAR
