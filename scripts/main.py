import sys
import os 
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_path, "..")
sys.path.append(root_dir)

from data.data_loader import get_dataloader
from utils.visualization import showing_samples

def load_data(data_dir):
    with open(data_dir, "rb") as f :
        data=pickle.load(f, encoding = "latin1")
    return data

def main():

    data_dir = os.path.join(root_dir, "data/dataset/mnist.pkl")
    data = load_data(data_dir=data_dir)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_dataloader(data)

    showing_samples(X_train, y_train)

    

if __name__ == "__main__":
    main()
