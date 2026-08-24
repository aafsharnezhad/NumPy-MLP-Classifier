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


# class OutPutLayer():
#     def __init__(self, A_prev, num_classes):
#         input_dim = A_prev.shape()[1]
#         self.w = np.random.rand(input_dim, neurons)
#         self.b = np.zeros(shape=(1, neurons))
#         self.dw = 0
#             self.dz = 0
    
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
    def __init__(self,A_prev, neurons):

        self.neurons = neurons
        input_dim = A_prev.shape()[1]
        self.W = np.random.rand(input_dim, neurons)
        self.b = np.zeros(shape=(1, neurons))
        self.dw = 0
        self.dz = 0
        self.batch_size = 64
        self.cache = {}
        self.V_dW = np.zeros_like(self.W)
        self.V_db = np.zeros_like(self.b)

    def forward(self, A_prev):
        z = A_prev @ self.W + self.b # [B, first_layer]
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
        self.second_layer_neurons = layers[1]
        self.num_classes = num_classes
        self.w1 = np.random.randn(784, self.first_layer_neurons)
        self.b1 = np.zeros(shape=(1,self.first_layer_neurons))
        self.Y_true = Y_ture
        self.X = X

        if layers[1] is not None :
            second_layer_neurons = layers[1]
            self.w2 = np.random.randn(self.first_layer_neurons, second_layer_neurons)
            self.b2 = np.zeros(shape=(1,second_layer_neurons))
            

        else : 
            second_layer_neurons = layers[0]

                

        self.last_w = np.random.rand(second_layer_neurons, num_classes)
        self.last_b = np.zeros(shape=(1,num_classes))
            

            
    def forward(self, X):

        self.layer_1 = DenseLayer(A_prev = X, neurons = self.first_layer_neurons)
        self.A_1, self.Z1 = self.layer_1.forward(A_prev = X)
        self.layer_2 = DenseLayer(A_prev = self.A_1, neurons = self.second_layer_neurons)
        self.A_2, self.Z2 = self.layer_2.forward(A_prev = self.A_1)
        self.out_put_layer = DenseLayer(A_prev = self.A_2, neurons = self.num_classes)
        self.y_predicted = self.out_put_layer.forward(A_prev = self.A_2)
        
        
        return self.y_predicted  


    def backward(self):
        dz_last, dw_last = self.out_put_layer.backward(self,  A_prev = self.A_2, Y_true = self.Y_true, Y_predicted = self.y_predicted)
        dz_2, dw_2 =  self.out_put_layer.backward(self, A_prev = self.A_1, w_next = self.last_w, current_z = self.Z2, dz_next = dz_last)
        dz_1, dw_1 =  self.out_put_layer.backward(self, A_prev = self.X, w_next = self.w2, current_z = self.Z1, dz_next = dz_2)
        return dw_last, dw_2, dw_1