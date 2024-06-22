###########################################
# EXPERIMENTS 1: BASELINE
# Non-state based: sample datapoints
###########################################

# General imports
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folderpath = os.path.join(os.getcwd(), 'acs_data')

# Import own functions
from data_loader import split_quantity_equal, load_data, arange_client_sizes, original_split, split_quantity_nonequal, get_state_metadata, split_quantity_binary, split_quantity_interval
from central_acs import train, test
from fl_pipeline import run_fl
from all_models import LogisticRegression, MLP, Linear, BinaryClassifier

###########################################
# Hyperparameters
###########################################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLIENTS = 50
CLIENT_SIZES = [36000]
model = LogisticRegression(9).to(DEVICE)
LEARNING_RATE = 0.001
NUM_ROUNDS = 5
EPOCHS = 5
FRACTION_FIT=1 # Sample 100% of available clients for training
FRACTION_EVALUATE=1 # Sample 50% of available clients for evaluation
SEEDS = [
    1042, 2753, 3291, 4856, 5178, 6920, 7482, 8391, 9023, 1038,
    1194, 1567, 2023, 2468, 2973, 3456, 3984, 4231, 4598, 5001,
    5234, 5648, 6091, 6472, 6890, 7345, 7689, 8123, 8574, 9021,
    9483, 1025, 1189, 1357, 1568, 1784, 1962, 2093, 2345, 2589,
    2750, 2981, 3145, 3298, 3452, 3789, 4032, 4268, 4591, 4876
]

###########################################
# get device and txt-file ready
###########################################

CLIENT_RESOURCES = {"num_cpus": 3, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    # here we are assigning an entire GPU for each client.
    CLIENT_RESOURCES = {"num_cpus": 3, "num_gpus": 1.0}

path = 'results/results_experiments_100.txt'

with open(path, 'w') as file:
    file.write('')

###########################################
# Train and evaluation on multiple seeds
###########################################

scores_ex1_acc = []
scores_ex1_eo = []
scores_ex1_dp = []
scores_ex2_acc = []
scores_ex2_eo = []
scores_ex2_dp = []
scores_ex3_acc = []
scores_ex3_eo = []
scores_ex3_dp = []
scores_ex4_acc = []
scores_ex4_eo = []
scores_ex4_dp = []
scores_ex5_acc = []
scores_ex5_eo = []
scores_ex5_dp = []

for SEED in SEEDS:
    ###########################################
    # Obtain data
    ###########################################
    print('\n Exp 1 ...\n')
    ex1_clients = split_quantity_equal(51, [36000], data_folderpath, SEED)
    ex1_trainloaders, ex1_valloaders, ex1_testloaders, ex1_complete_trainloader, ex1_complete_valloader, ex1_complete_testloader = load_data(ex1_clients, SEED, batch_size=32)
    ex1_train_samples = train(model, ex1_complete_trainloader, epochs=EPOCHS, lr=LEARNING_RATE)
    ex1_loss, ex1_acc, ex1_eo, ex1_dp, ex1_incorrect, ex1_correct = test(model, ex1_complete_testloader)
    scores_ex1_acc.append(ex1_acc)
    scores_ex1_eo.append(ex1_eo)
    scores_ex1_dp.append(ex1_dp)

    print('\n Exp 2 ...\n')
    ex2_client_sizes = arange_client_sizes(320, 100000, 51, 1736000)
    ex2_clients = split_quantity_nonequal(NUM_CLIENTS, ex2_client_sizes, data_folderpath, SEED)
    ex2_trainloaders, ex2_valloaders, ex2_testloaders, ex2_complete_trainloader, ex2_complete_valloader, ex2_complete_testloader = load_data(ex2_clients, SEED, batch_size=32)
    ex2_train_samples = train(model, ex1_complete_trainloader, epochs=EPOCHS, lr=LEARNING_RATE)
    ex2_loss, ex1_acc, ex1_eo, ex1_dp, ex1_incorrect, ex1_correct = test(model, ex1_complete_testloader)
    scores_ex2_acc.append(ex2_acc)
    scores_ex2_eo.append(ex2_eo)
    scores_ex2_dp.append(ex2_dp)

    print('\n Exp 3 ...\n')
    ex3_clients = original_split()
    ex3_trainloaders, ex3_valloaders, ex3_testloaders, ex3_complete_trainloader, ex3_complete_valloader, ex3_complete_testloader = load_data(ex3_clients, SEED, batch_size=32)
    ex3_train_samples = train(model, complete_trainloader, epochs=EPOCHS, lr=LEARNING_RATE)
    ex3_loss, ex3_acc, ex3_eo, ex3_dp, ex3_incorrect, ex3_correct = test(model, ex3_complete_testloader)
    scores_ex3_acc.append(ex3_acc)
    scores_ex3_eo.append(ex3_eo)
    scores_ex3_dp.append(ex3_dp)

    print('\n Exp 4 ...\n')
    ex4_metadata = get_state_metadata()
    ex4_clients = split_quantity_binary(ex4_metadata, int(30/2))
    ex4_trainloaders, ex4_valloaders, ex4_testloaders, ex4_complete_trainloader, ex4_complete_valloader, ex4_complete_testloader = load_data(ex4_clients, SEED, batch_size=32)
    ex4_train_samples = train(model, ex4_complete_trainloader, epochs=EPOCHS, lr=LEARNING_RATE)
    ex4_loss, ex4_acc, ex4_eo, ex4_dp, ex4_incorrect, ex4_correct = test(model, ex4_complete_testloader)
    scores_ex4_acc.append(ex4_acc)
    scores_ex4_eo.append(ex4_eo)
    scores_ex4_dp.append(ex4_dp)

    print('\n Exp 5 ...\n')
    ex5_metadata = get_state_metadata()
    ex5_clients = split_quantity_interval(ex5_metadata, int(30/2))
    ex5_trainloaders, ex5_valloaders, ex5_testloaders, ex5_complete_trainloader, ex5_complete_valloader, ex5_complete_testloader = load_data(ex5_clients, SEED, batch_size=32)
    ex5_train_samples = train(model, ex5_complete_trainloader, epochs=EPOCHS, lr=LEARNING_RATE)
    ex5_loss, ex5_acc, ex5_eo, ex5_dp, ex5_incorrect, ex5_correct = test(model, ex5_complete_testloader)
    scores_ex5_acc.append(ex5_acc)
    scores_ex5_eo.append(ex5_eo)
    scores_ex5_dp.append(ex5_dp)

ex1_stdev, ex1_mean = calculate_stdev(scores_ex1_acc, scores_ex1_eo, scores_ex1_dp)
ex2_stdev, ex2_mean = calculate_stdev(scores_ex2_acc, scores_ex2_eo, scores_ex2_dp)
ex3_stdev, ex3_mean = calculate_stdev(scores_ex3_acc, scores_ex3_eo, scores_ex3_dp)
ex4_stdev, ex4_mean = calculate_stdev(scores_ex4_acc, scores_ex4_eo, scores_ex4_dp)
ex5_stdev, ex5_mean = calculate_stdev(scores_ex5_acc, scores_ex5_eo, scores_ex5_dp)
print('Experiment 1 mean:', ex1_mean)
print('Experiment 1 mean:', ex1_stdev)
print('--------------')
print('Experiment 2 mean:', ex2_mean)
print('Experiment 2 mean:', ex2_stdev)
print('--------------')
print('Experiment 3 mean:', ex3_mean)
print('Experiment 3 mean:', ex3_stdev)
print('--------------')
print('Experiment 4 mean:', ex4_mean)
print('Experiment 4 mean:', ex4_stdev)
print('--------------')
print('Experiment 5 mean:', ex5_mean)
print('Experiment 5 mean:', ex5_stdev)
print('--------------')
