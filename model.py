import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import itertools
import torchmetrics

#First, we define the classes for a transformation layer in the networks of the clients. Each network learns
#the parameters of a given transformation.

#To compare, we define a layer without transformation.
class withoutTransformation(nn.Module):
    def __init__(self, in_features, out_features):
        super(withoutTransformation, self).__init__()
    
    def forward(self, x):
        return x

#Defines a linear transformation layer
class LinearTransformation(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearTransformation, self).__init__()
        self.alpha = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.mul(x, self.alpha) + self.beta

#Defines a tanh transformation layer
class TanhTransformation(nn.Module):
    def __init__(self, in_features, out_features):
        super(TanhTransformation, self).__init__()
        self.alpha = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        return torch.tanh(torch.mul(x, self.alpha) + self.beta)

#The following class defines a network class for clients. Note that n_entradas is the input size.
class Net(nn.Module):
    def __init__(self, transformation_type):
        super(Net, self).__init__()
        
        #Defines the transformation           
        if transformation_type == "LinearTransformation":
            self.transformation = LinearTransformation(n_entradas, n_entradas)
        elif transformation_type == "TanhTransformation":
            self.transformation = TanhTransformation(n_entradas, n_entradas)
        elif transformation_type == "withoutTransformation":
            self.transformation = withoutTransformation(n_entradas, n_entradas)
            
        #Defines other layers
        self.fc1 = nn.Linear(n_entradas, 50)
        self.fc2 = nn.Linear(50, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.transformation(x)
        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x)) 
        x = self.fc3(x)    
        return x

#Defines a network class for the Server.
class Net_Server(nn.Module):
    def __init__(self):
        super(Net_Server, self).__init__()
        self.fc1 = nn.Linear(n_entradas, 50)
        self.fc2 = nn.Linear(50, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x)) 
        x = self.fc3(x)    
        return x

#Creates a client class, initiates a privacy engine and trains the network.
class Client:
    def __init__(self, data, target, client_number, transformation, epsilon):
        self.transformation_type = transformation
        self.model = self._init_model()
        self.client_number = client_number
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCEWithLogitsLoss()
        
        dataset = TensorDataset(data, target.view(-1, 1))
        self.data_loader = DataLoader(dataset, batch_size=25, shuffle=True)
        #Initiates the privacy engine for Differential privacy.
        self.privacy_engine = PrivacyEngine()  
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            epochs=8,
            target_epsilon=epsilon,
            target_delta=0.001,
            max_grad_norm=1,
        )
    def _init_model(self):
        model = Net(self.transformation_type)
        return model
    #Trains the neural network.
    def train(self, global_model, transformation, epsilon, federated_round, epochs=8):
        self.model.train()
        
        #Creates and extended local model by downloading the global model and keeping the 
        #transformation from previous federated round.
      
        transformations = self.model.state_dict() if self.model.state_dict() else {
            "_module.transformation.alpha": torch.ones(n_entradas),
            "_module.transformation.beta": torch.zeros(n_entradas)}
        global_dict = global_model.state_dict()
        common_keys = global_dict.keys() & transformations.keys()
        transformations.update({k: global_dict[k] for k in common_keys})
        self.model.load_state_dict(transformations)

        #Trains the neural network
        for epoch in range(epochs):
            total_correct = 0
            total_samples = 0
            for data, target in self.data_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.float())
                loss.backward()
                self.optimizer.step()
              
        return self.model.state_dict()

#Creates a Server and aggregates local models.
class Server:
    def __init__(self):
        self.global_model = Net_Server()
        self.criterion = nn.BCEWithLogitsLoss()  # Defining the loss function
    #Server does the Federated Learning steps. Does the aggregation using FedAVG.
    def aggregate(self, client_models):
        global_dict = {}
        for k in client_models[0].keys():
            adjusted_key = k.replace("_module.", "")
            if adjusted_key not in ["transformation.alpha", "transformation.beta"]:
                global_dict[adjusted_key] = torch.stack([client_models[i][k] for i in range(len(client_models))], 0).mean(0)
        self.global_model.load_state_dict(global_dict)

#Runs the model. 
eps = 0.1
trans = "TanhTransformation"
federated_rounds = 10
N = 5 #Number of clients
clients = [Client(client_data[i], client_target[i], i, trans, eps) for i in range(N)]
server = Server()
for i in federated_rounds:
  print(f"Federated round: {i}, Transformation: {trans}, Objective Epsilon: {eps}")
  client_models = [client.train(server.global_model, trans, eps, i) for client in clients]
  server.aggregate(client_models)
