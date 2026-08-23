import numpy as np 


def relu(x):

    if x >=0:
        return x
    else :
        return 0   

def soft_max(Z): 
    
    
    predictions = np.exp(Z) / np.sum(np.exp(Z), axix = 1, keepdims=True)

    predicted_class = np.argmax(predictions)
    return predicted_class

    
    
    


class FeedForwardNeuralNetwork():
    def __init__(self,layers ,batch_size = 32, num_classes = 10):
        first_layer_neurons = layers[0]
        self.w1 = np.random.randn(784, first_layer_neurons)
        self.b1 = np.zeros(shape=(1,first_layer_neurons))

        if layers[1] is not None :
            second_layer_neurons = layers[1]
            self.w2 = np.random.randn(first_layer_neurons, second_layer_neurons)
            self.b2 = np.zeros(shape=(1,second_layer_neurons))
            

        else : 
            second_layer_neurons = layers[0]

                

        self.last_w = np.random.rand(second_layer_neurons, num_classes)
        self.last_b = np.zeros(shape=(1,num_classes))
            

            
    def forward(self, X):
        
        # Xw + b = z(1)
        # Z(1)w + b = z(2)

        z_1 = X @ self.w1 + self.b1 # [B, first_layer]
        a_1 = relu(z_1) # [B, first_layer]
        if self.w2 is not None :
            z_2 = a_1 @ self.w2 + self.b2 # [B, second_layer]
            a_2 = relu(z_2)
            
            last_z = a_2 @ self.last_w + self.last_b #[B, num_classes]
            predicted_class = soft_max(last_z)
        else :
            last_z = a_1 @ self.last_w + self.last_b #[B, num_classes]
            predicted_class = soft_max(last_z)

        return predicted_class    


    def backward(self):
        pass