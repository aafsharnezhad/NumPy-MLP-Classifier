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


  
def cross_entropy_loss(y_pred, y_true):
    # Extracting Batch Size
    B = y_pred.shape[0]

    epsilon = 1e-4
    np.clip(y_pred)

    ## The Loss will be a (B * 10) matrix 
    loss = -1/B * (np.sum(y_true * (np.log(y_pred))))
    return loss    


def compute_total_loss(): 
    pass   

def relu_backward(x):
    if x>0 : 
        return 1
    else : 
        return 0 



class DenseLayer():
    def __init__(self,input_dim, neurons):

        self.neurons = neurons
        
        self.W = np.random.rand(input_dim, neurons)
        self.b = np.zeros(shape=(1, neurons))
        self.dw = 0
        self.dz = 0
        
        self.cache = {}
        self.V_dW = np.zeros_like(self.W)
        self.V_db = np.zeros_like(self.b)

    def forward(self, A_prev):
        z = A_prev @ self.W + self.b # [B, first_layer]
        self.cache['current_z'] = z
        self.cache['A_prev'] = A_prev 
        return z 

    def backward(self, learning_rate, w_next=None, beta = 0.9, current_z=None, Y_true=None, Y_predicted=None, dz_next=None, lambda_reg=0):
        A_prev = self.cache["A_prev"]


        if self.neurons == 10 : 
            m = A_prev.shape[0]
            self.dz = (Y_true - Y_predicted)/m
        else : 
            self.dz = dz_next * w_next * relu_backward(current_z)

        
        self.dw = self.dz * A_prev
        self.db = np.sum(self.dz, axis=0, keepdims=True)

        
        self.dw = self.dz * A_prev
        self.db = np.sum(self.dz, axis=0, keepdims=True)

        # without lambda_reg ---> g(t) = grad(L(w)) ||| with lambda_reg ---> g(t) = grad(L(w) + ((lambda/m) * w))
        if lambda_reg>0 :
            self.dw += ((lambda_reg/m) * self.W)


        self.dA_prev = self.dz @ self.W    

        self.V_dW = (beta * self.V_dW) + ((1 - beta) * self.dW)
        self.V_db = (beta * self.V_db) + ((1 - beta) * self.db)

        self.W -= learning_rate * self.V_dW
        self.b -= learning_rate * self.V_db

        return self.dz
        


class FeedForwardNeuralNetwork():
    def __init__(self,layers, X, Y_ture, batch_size = 32, num_classes = 10):
        self.first_layer_neurons = layers[0]


        self.y_true = Y_ture 
        self.num_classes = num_classes

        self.model_layers = []


        ### example : layers = [16,32] ----> model_layers = [DenseLayer, Dense_layer, Dense_layer]
        for i in range(len(layers)) : 
            if i == 0 :
                input_dim = 784
            else : 
                input_dim = layers[i-1]    

            self.model_layers.append(DenseLayer(input_dim=input_dim , neurons = layers[i]))
        # Creating Last hidden Layer    
        self.model_layers.append(DenseLayer(input_dim=layers[-1], neurons=num_classes))



        # self.w1 = np.random.randn(784, self.first_layer_neurons)
        # self.b1 = np.zeros(shape=(1,self.first_layer_neurons))


        self.Y_true = Y_ture

        self.X = X

        # if len(layers) > 2 :
        #     second_layer_neurons = layers[1]
        #     # self.w2 = np.random.randn(self.first_layer_neurons, second_layer_neurons)
        #     # self.b2 = np.zeros(shape=(1,second_layer_neurons))
            

        # else : 
        #     second_layer_neurons = layers[0]

                

        # self.last_w = np.random.rand(second_layer_neurons, num_classes)
        # self.last_b = np.zeros(shape=(1,num_classes))
            

            
    def forward(self, X):

        A_prev = X
        for i in range(len(self.model_layers) - 1):
           layer =  self.model_layers[i]
           Z = layer.forward(A_prev = A_prev)
           A = relu(Z)
           A_prev = A

        layer = self.model_layers[-1]
        Z_out = layer.forward(A_prev = A_prev)
        A_out = soft_max(Z_out)   
        
        self.y_predicted = A_out
         
        
        return self.y_predicted  


    def backward(self, learning_rate):
        for i in reversed(range(len(self.model_layers)-1)):
            layer = self.model_layers[i]
            
            current_z = layer.cache['current_z']
            dZ = layer.backward(learning_rate, w_next=w_next, beta = 0.9, current_z=current_z, Y_true=self.y_true, Y_predicted=self.y_predicted, dz_next=dz_next, lambda_reg=0)
            w_next = layer.W
            dz_next = dZ
