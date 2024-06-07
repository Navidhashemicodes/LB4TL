import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import time
import sys
import pathlib

RP_dir = str (pathlib.Path().resolve())
RP_dir = RP_dir + '/../../../..'  #'C:/Users/navid/Documents/MATLAB/MATLAB_prev/Toyota/ADHS_RP'

FOLDERNAME = RP_dir
sys.path.append( FOLDERNAME + '/Test_Pytorch' )
from functions import export2matlab

###################################Accurate

class RNetwork_acc(nn.Module):
    def __init__(self, weights_file, biases_file):
        super(RNetwork_acc, self).__init__()

        # Load weights and biases from MATLAB files
        weights_Linear_mat = loadmat(weights_file)['WL']
        biases_Linear_mat = loadmat(biases_file)['BL']
        weights_nLinear_mat = loadmat(weights_file)['Wn']
        biases_nLinear_mat = loadmat(biases_file)['Bn']

        # Convert cell array elements to PyTorch tensors with type conversion
        self.weights_L = [torch.from_numpy(wi.astype(np.float32)) for wi in weights_Linear_mat[0]]
        self.weights_n = [torch.from_numpy(wi.astype(np.float32)) for wi in weights_nLinear_mat[0]]
        self.biases_L = [torch.from_numpy(bi.astype(np.float32)) for bi in biases_Linear_mat[0]]
        self.biases_n = [torch.from_numpy(bi.astype(np.float32)) for bi in biases_nLinear_mat[0]]

        # Calculate the number of hidden layers based on the length of weights
        num_hidden_layers = len(self.weights_L) - 1

        # Define the hidden layers using predefined weights and biases
        self.hidden_layers = nn.ModuleList([
            nn.Linear( (self.weights_n[i]).shape[1], (self.weights_n[i]).shape[0])
            for i in range(num_hidden_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear((self.weights_L[-1]).shape[1], (self.weights_L[-1]).shape[0])
        
        
        for i, layer in enumerate(self.hidden_layers):
            layer.weight.data = self.weights_n[i]  # Transpose for proper shape
            layer.bias.data = self.biases_n[i]

        self.output_layer.weight.data = self.weights_L[-1]
        self.output_layer.bias.data = self.biases_L[-1]

    def forward(self, x):
        # Forward pass through hidden layers with ReLU activation
        for i, layer in enumerate(self.hidden_layers):
            xn = nn.functional.relu(layer(x))
            xl = torch.matmul( x , (self.weights_L[i]).t()  ) + self.biases_L[i]
            x=[]
            x.append(xl)
            x.append(xn)
            x = torch.cat(x , dim=1) 
            
        # Output layer
        x = self.output_layer(x)
        return x




###########################################smooth
def boltzmann(theinput, beta):
    m = nn.Softmax(dim=0)
    y = m(beta * theinput)
    z = theinput * y
    return torch.sum(z).flatten()

def softmin(x, beta):
    result = -torch.logsumexp(-beta * x, dim=0) / beta
    return result.flatten()

def h_g1(s, scale, beta):
    hs = torch.zeros(4)
    hs[0] = scale * 6 - s[0]
    hs[1] = s[0] - scale * 5
    hs[2] = s[1] - scale * 3
    hs[3] = scale * 4 - s[1]
    return softmin(hs, beta)

def h_g2(s, scale, beta):
    hs = torch.zeros(4)
    hs[0] = scale * 4 - s[0]
    hs[1] = s[0] - scale * 3
    hs[2] = s[1]
    hs[3] = scale * 1 - s[1]
    return softmin(hs, beta)

def h_g3(s, scale, beta):
    hs = torch.zeros(4)
    hs[0] = scale * 1 - s[0]
    hs[1] = s[0] - scale * 4
    hs[2] = s[1] - scale * 5
    hs[3] = scale * 2 - s[1]
    return boltzmann(hs, beta)

def robustness(x, scale):
    
    T = 40
    T = T+1
    Beta = 10
    rob0 = torch.zeros(T-1)
    for i in range(1, T):
        rob1 = torch.zeros(i)
        rob2 = torch.zeros(T-i)
        for j in range(0, i):
            rob1[j] = h_g1(x[j, :], scale, beta=Beta)
        
        for j in range(i, T):
            rob2[j-i] = h_g2(x[j, :], scale, beta=Beta)
        
        B1 = boltzmann(rob1, beta=Beta)
        
        B2 = boltzmann(rob2, beta=Beta)
        
        rob0[i-1] = softmin(torch.cat([B1, B2]), beta=Beta)
        
    

    Rob1 = boltzmann(rob0, beta=Beta)
    

    rob0 = torch.zeros(T)
    for i in range(0, T):
        rob0[i] = h_g3(x[i, :], scale, beta=Beta)
    
    Rob2 = softmin(rob0, beta=Beta)
        
    rho = softmin(torch.cat([Rob1, Rob2]), beta=Beta)
   

    return rho


def model(s, a):

    dt = 0.05
    L = 1

    a = 0.5 * a

    v = 2.5 * torch.tanh(a[0, 0]) + 2.5
    gam = (torch.pi / 4) * torch.tanh(a[0, 1])

    f1 = s[0, 0] + (L / torch.tan(gam)) * (torch.sin(s[0, 2] + (v / L) * torch.tan(gam) * dt) - torch.sin(s[0, 2]))
    f2 = s[0, 1] + (L / torch.tan(gam)) * (-torch.cos(s[0, 2] + (v / L) * torch.tan(gam) * dt) + torch.cos(s[0, 2]))
    f3 = s[0, 2] + (v / L) * torch.tan(gam) * dt

    s_next = torch.stack([f1, f2, f3]).unsqueeze(0)

    return s_next






# Initialize the networks
input_size  = 4
hidden_size = [10]
output_size = 2
num_epochs = 100000
T = 40
scale = torch.tensor(1, dtype=torch.float32)

##############################################

initweights_file_path = FOLDERNAME + '/Test_Pytorch/comparison_vehicle/5inits/init4/init_weights.mat'
initbiases_file_path = FOLDERNAME + '/Test_Pytorch/comparison_vehicle/5inits/init4/init_biases.mat'

weights_mat = loadmat(initweights_file_path)['W']
biases_mat = loadmat(initbiases_file_path)['b']
        
weights = [torch.from_numpy(wi.astype(np.float32)) for wi in weights_mat[0]]
biases = [torch.from_numpy(bi.astype(np.float32)) for bi in biases_mat[0]]

initial_params = {}
for i in range(len(weights)):
    initial_params[f'{2 * i}.weight'] = weights[i]
    BB = biases[i]
    initial_params[f'{2 * i}.bias'] = BB.flatten()

controller_net = nn.Sequential(
    nn.Linear(input_size, hidden_size[0]),
    nn.ReLU(),
    nn.Linear(hidden_size[0], output_size)
)


# for layer in controller_net:
#     if isinstance(layer, nn.Linear):
#         init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='relu')


controller_net.load_state_dict(initial_params)

# export2matlab("Controller0",controller_net)
######################################


# Define the optimizer for f_network
optimizer = optim.Adam(controller_net.parameters(), lr=0.001)



weights_file_path_acc = FOLDERNAME + '/Test_Pytorch/comparison_vehicle/5inits/init4/weights_acc.mat'
biases_file_path_acc = FOLDERNAME + '/Test_Pytorch/comparison_vehicle/5inits/init4/biases_acc.mat'

r_network_acc = RNetwork_acc(weights_file_path_acc, biases_file_path_acc)



max_time_seconds = 3600

# Record the start time
start_time = time.time()

torch.manual_seed(0)

candidates = [-6, -5, -4]
for epoch in range(num_epochs):
    # Initialize the state at time t=1
    selected_candidate_index = torch.randint(len(candidates), (1,))
    selected_candidate = candidates[selected_candidate_index.item()]
    # state = torch.zeros((1, input_size), requires_grad=True)
    state = torch.tensor([[6*scale, 8*scale , selected_candidate*torch.pi/8 ]], requires_grad=True)+0.0005*torch.rand([1,3])
    # Forward pass through the dynamic system and collect the trajectory
    trajectory = []
    trajectory.append(state)
    for t in range(0, T):
        Time = (torch.tensor(t, dtype=torch.float32)).unsqueeze(0).unsqueeze(1)
        sa = torch.cat([state, Time], dim=1)
        state = model( state, controller_net(sa) )
        trajectory.append(state)

    # Convert the trajectory to a tensor
    trajectory = torch.cat(trajectory, dim=0)
    # print(trajectory)
    # Forward pass through the modified objective function
    objective_value = robustness(trajectory , scale)
    
    
    
    if epoch % 50 == 0:
        objective_value_acc = torch.zeros([1,3])
        for i in range(0,3):
            state_check = torch.tensor([[6*scale, 8*scale , candidates[i]*torch.pi/8 ]], requires_grad=False)+0.0005*torch.rand([1,3])
            trajectory_check = []
            trajectory_check.append(state_check)
            for t in range(0, T):
                time_check = (torch.tensor(t, dtype=torch.float32)).unsqueeze(0).unsqueeze(1)
                sa_check = torch.cat([state_check, time_check], dim=1)
                state_check = model( state_check, controller_net(sa_check) )
                trajectory_check.append(state_check)

            # Convert the trajectory to a tensor
            trajectory_check = torch.cat(trajectory_check, dim=1)
        
            # Forward pass through the modified objective function
            objective_value_acc[0,i] = r_network_acc(trajectory_check)
 
        decision = torch.min( objective_value_acc , dim=1)
        print(f'Epoch {epoch + 1}, Objective Value: {objective_value.item()}, obj accurate: {objective_value_acc}' )

        # Check if the objective value is positive
        if decision.values > 0:
            print(decision)
            elapsed_time = time.time() - start_time
            print(f'Terminating process at Epoch {epoch + 1} after ( {elapsed_time} ) seconds, because objective value is positive.')
            break
    
    

    # Backward pass and optimization to maximize the objective function
    optimizer.zero_grad()
    (-objective_value).backward()
    optimizer.step()

    # Print the objective value during training
    print(f'Epoch {epoch + 1}, Objective Value: {objective_value.item()}')

    elapsed_time = time.time() - start_time
    if elapsed_time > max_time_seconds:
        print("Time limit exceeded. Breaking out of the loop.")
        break
    
    



export2matlab("RP_reviewer_Controller_init4_Kurtz",controller_net)
savemat('training_info.mat', {'Runtime': elapsed_time , 'rho_min': decision.values.detach().numpy().astype(float)})
