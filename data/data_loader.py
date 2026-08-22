 
import os 
import pickle
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_path, "..")

def load_data(data_dir):
    with open(data_dir, "rb") as f :
        data=pickle.load(f, encoding = "latin1")
    return data



def get_dataloader(data):

    
    train_loader = data[0]
    val_loader = data[1]
    test_loader = data[2]


    return train_loader, val_loader, test_loader    


def get_data_size(): 

    data_dir = os.path.join(root_dir, "data/dataset/mnist.pkl")
    data = load_data(data_dir) 

    ### data = ((x_train, y_train), (x_val, y_val), (x_test, y_test)) 
    print("Train_size : ", len(data[0][0]))
    print("Val Size : ", len(data[1][0]))
    print("Test Size : ", len(data[2][0]))


def standardize_data(data):

    epsilon = 1e-8
    x_train = data[0][0]
    x_val = data[1][0]
    x_test = data[2][0]

    x_train = np.array(x_train)
    flatted_x_train = x_train.flatten()
    mean_train = np.mean(flatted_x_train, axis = 0)
    std_train = np.std(flatted_x_train, axis = 0)

    normalized_x_train = (x_train - mean_train)/(std_train + epsilon)  
    normalized_x_val = (x_val - mean_train)/(std_train + epsilon) 
    normalized_x_test = (x_test - mean_train)/(std_train + epsilon)

    return normalized_x_train, normalized_x_val, normalized_x_test


    





    



    