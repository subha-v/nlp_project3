
from model import *
from utils import *
from data import *


class NameGenerator:
    def __init__(self, dataset_path, num_iters, save_model_path):
        self.data = NamesDataset(dataset_path)

        self.model = RNN_GEN(self.data.num_countries, NUM_LETTERS, NUM_LETTERS) #we want the output to be a bunch of probabilities so the highest value is the next letter
        self.loss_func = nn.NLLLoss()
        self.learning_rate = 0.0005
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)
        self.num_iters = num_iters
        self.n_print = 5000
        self.max_name_length = 15 #the max number of letters allowed
        self.save_model_path = save_model_path

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_model_path) #saving all the weights and stuff in the model

    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_model_path)) #loading out weights and biases



    def train_step(self, name_tensor, country_tensor, target_tensor):
        num_letters = name_tensor.size()[0] #shows how long the letters is
        loss = 0 
        hidden_state = self.model.init_hidden()

        #now we need to iterate over all the letters and 
        for i in range(num_letters):
            #Call the Model, pass in the inputs and then get the outputs
            output, hidden_state = self.model(name_tensor[i], country_tensor, hidden_state)
            loss += self.loss_func(output,target_tensor[:, i]) #calling the loss function on these values
            #we need to index into the second dimension which contains the length of the name instead of hte first dimension which is just 1
            #we need to look at the probability of the i'th letter
            #add up the loss for all of the letters
            self.optimizer.zero_grad() # we need to Zero things out otherwies they accumulate
            loss.backward()
            self.optimizer.step()

            return output, loss.item()/num_letters #WE NEED TO FIND THE AVERAGE LOSS over all letters instead of the sum
            
    def train(self):
        for i in range(self.num_iters):
            country, name, country_tensor, name_tensor, target_tensor = self.data.get_random_sample()
            output, loss = self.train_step(name_tensor, country_tensor, target_tensor)

            if (i%self.n_print == 0):
                print(f"Iter {i+1}: Loss  {loss:.4f}")
                print("Neural Network Generated Names: ")
                self.print_sample_names()
                print("-"*40)

    def sample_neural_network(self, country, starting_letter):
        starting_letter = starting_letter.lower()
        with torch.no_grad():
            country_tensor = self.data.country_to_tensor(country)
            input_letter_tensor = letter_to_tensor(starting_letter)
            hidden_state = self.model.init_hidden() #initializing the hidden state
            #We need to iterate over the max number of letters
            #each iteration we need to call the model
            # if the letter is not the end charcter, then add it to the name we are building
            #otherwise exit
            output_name=starting_letter

            for i in range(0,self.max_name_length-1):
                output, hidden_state = self.model(letter_to_tensor(output_name[i]), country_tensor, hidden_state)
                output_letter = self.data.output_to_letter(output)
                output_name+=output_letter


                if(output_letter==END_CHAR):
                    break
        
        return output_name

    def print_sample_names(self):
        print(self.sample_neural_network('Indian','C'))
        print(self.sample_neural_network('Indian','A'))
        print(self.sample_neural_network('Indian','H'))

        print(self.sample_neural_network('Korean','K'))
        print(self.sample_neural_network('Korean','S'))
        print(self.sample_neural_network('Korean','C'))
        print(self.sample_neural_network('Korean','L'))



            
if __name__ == '__main__':
    name_generator = NameGenerator("data/names/*.txt", 10000, "model.pth")
    # name_generator.train()
   # name_generator.save_model()
    name_generator.load_model()
    
    print('Final Sampling: ')
    name_generator.print_sample_names()

          








        #target tensor is the  



