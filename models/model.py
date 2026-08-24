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
        self.w = np.random.rand(input_dim, neurons)
        self.b = np.zeros(shape=(1, neurons))
        self.dw = 0
        self.dz = 0
        self.batch_size = 64

    def forward(self, A_prev):
        z = A_prev @ self.w1 + self.b1 # [B, first_layer]
        if self.neurons == 10:
            a = soft_max(z)
        else :
            a = relu(z)# [B, first_layer]
        return a 

    def backward(self, A_prev = None, w_next = None, current_z = None, Y_true = None, Y_predicted = None, dz_next = None):
        if self.neurons == 10:
            self.dz = (Y_true - Y_predicted)/self.batch_size
            self.dw = self.dz * A_prev
            return self.dz
        else:

            self.dz = dz_next * w_next * relu_backward(current_z)
            self.dw = self.dz * A_prev
            return self.dz
        


class FeedForwardNeuralNetwork():
    def __init__(self,layers ,batch_size = 32, num_classes = 10):
        self.first_layer_neurons = layers[0]
        self.second_layer_neurons = layers[1]
        self.num_classes = num_classes
        self.w1 = np.random.randn(784, self.first_layer_neurons)
        self.b1 = np.zeros(shape=(1,self.first_layer_neurons))


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
        self.A_1 = self.layer_1.forward(A_prev = X)
        self.layer_2 = DenseLayer(A_prev = self.A_1, neurons = self.second_layer_neurons)
        self.A_2 = self.layer_2.forward(A_prev = self.A_1)
        self.out_put_layer = DenseLayer(A_prev = self.A_2, neurons = self.num_classes)
        self.y_predicted = self.out_put_layer.forward(A_prev = self.A_2)
        
        
        


        # # Xw + b = z(1)
        # # Z(1)w + b = z(2)

        # z_1 = X @ self.w1 + self.b1 # [B, first_layer]
        # a_1 = relu(z_1) # [B, first_layer]
        # if self.w2 is not None :
        #     z_2 = a_1 @ self.w2 + self.b2 # [B, second_layer]
        #     a_2 = relu(z_2)
            
        #     last_z = a_2 @ self.last_w + self.last_b #[B, num_classes]
        #     predicted_class = soft_max(last_z)
        # else :
        #     last_z = a_1 @ self.last_w + self.last_b #[B, num_classes]
        #     predicted_class = soft_max(last_z)

        return self.y_predicted  


    def backward(self):
        dz_last = self.out_put_layer.backward(self,  A_prev = self.A_2, Y_true = None, Y_predicted = self.y_predicted)