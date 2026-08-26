 
import os 
import pickle
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_path, "..")



def get_dataloader(data):

    
    train_loader = data[0]
    val_loader = data[1]
    test_loader = data[2]


    return train_loader, val_loader, test_loader    


def get_data_information(data): 

    
    

    train_loader, val_loader, test_loader = get_dataloader(data)
    (x_train, y_train) = train_loader

    ### data = ((x_train, y_train), (x_val, y_val), (x_test, y_test)) 
    print("Train_size : ", len(x_train))
    print("Val Size : ", len(val_loader[0]))
    print("Test Size : ", len(test_loader[0]))

    print("Type of X_train : ", type(x_train))
    print("Shape of X_train : ",np.shape(x_train))


def standardize_data(data):


    train_loader = data[0]
    val_loader = data[1]
    test_loader = data[2]

    epsilon = 1e-8


    x_train = train_loader[0]
    x_val = val_loader[0]
    x_test = test_loader[0]
    y_train = train_loader[1]
    y_val = val_loader[1]
    y_test = test_loader[1]

    

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)    
    mean_train = np.mean(x_train, axis = 0)
    std_train = np.std(x_train, axis = 0)

    normalized_x_train = (x_train - mean_train)/(std_train + epsilon)  
    normalized_x_val = (x_val - mean_train)/(std_train + epsilon) 
    normalized_x_test = (x_test - mean_train)/(std_train + epsilon)

    return (normalized_x_train,y_train), (normalized_x_val, y_val), (normalized_x_test, y_test)


# get_data_information()



    



    