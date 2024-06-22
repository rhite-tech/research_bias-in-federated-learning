###########################################
# EXPERIMENTS 17: Feature: Sex, percentage
# without quantity heterogeneity
###########################################

# General imports
import torch
import sys
import os
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folderpath = os.path.join(os.getcwd(), 'acs_data')

# Import own functions
from data_loader import load_data, split_feature_percentage
from fl_pipeline import run_fl
from train_test import LogisticRegression, train, test

def main(args):
    ###########################################
    # Hyperparameters
    ###########################################
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLIENTS = args.num_clients
    model = LogisticRegression(9).to(DEVICE)
    LEARNING_RATE = args.learning_rate
    NUM_ROUNDS = args.num_rounds
    EPOCHS = args.epochs
    FRACTION_FIT = args.fraction_fit
    FRACTION_EVALUATE = args.fraction_evaluate
    PERC = args.perc
    SEEDS = args.seeds

    ###########################################
    # get device and txt-file ready
    ###########################################

    CLIENT_RESOURCES = {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus}
    if DEVICE.type == "cuda" and args.num_gpus > 0:
        CLIENT_RESOURCES = {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus}

    path = args.result_path

    with open(path, 'w') as file:
        file.write('')

    ###########################################
    # Train and evaluation on multiple seeds
    ###########################################

    for SEED in SEEDS:

        ###########################################
        # Obtain data
        ###########################################
        print('\n Start data processing \n')
        clients, len_clients = split_feature_percentage("Sex", data_folderpath, NUM_CLIENTS, PERC, SEED)
        trainloaders, valloaders, testloaders, complete_trainloader, complete_valloader, complete_testloader, missing = load_data(clients, SEED, batch_size=32, missing=True)

        ###########################################
        # Central Learning
        ###########################################
        print('\n Start running Central \n')
        train_samples = train(model, complete_trainloader, epochs=EPOCHS, lr=LEARNING_RATE)
        loss, metrics = test(model, complete_testloader)
        accuracy, eo, dp, max_min_classes = metrics['accuracy'], metrics['equalized_odds'], metrics['demographic_parity'], metrics["minmax_classes"]

        ###########################################
        # Federated Learning
        ###########################################
        print('\n Start running FL \n')
        fl_results = run_fl(model, trainloaders, testloaders, complete_trainloader, complete_testloader, DEVICE, len_clients, NUM_ROUNDS, FRACTION_FIT, FRACTION_EVALUATE, LEARNING_RATE)

        ###########################################
        # Write results out
        ###########################################

        with open(path, 'a') as file:
            file.write(f'Results Experiments 17 (seed {SEED}) \n')
            file.write('-'*40)
            file.write(f'\nModel: {type(model).__name__}')
            file.write(f'\nLearning Rate: {LEARNING_RATE} \n')
            file.write(f'#Train samples: {len(complete_trainloader)*32}, #test samples {len(complete_testloader)*32} \n')
            file.write(f'Missing classes: {missing} \n \n')
            file.write('-'*40)
            file.write('\nResults Central\n')
            file.write(f'Accuracy: {accuracy}\n')
            file.write(f"Equalized Odds {eo}\n")
            file.write(f"Demographic Parity {dp} \n")
            file.write(f"The max min classes for eo and dp:, {max_min_classes}")
            file.write('-'*40)
            file.write('\n Results Federated')
            file.write(str(fl_results))
            file.write('\n \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run federated learning experiments.')
    parser.add_argument('--num_clients', type=int, default=51, help='Number of clients')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of FL rounds')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--fraction_fit', type=float, default=1.0, help='Fraction of clients for training')
    parser.add_argument('--fraction_evaluate', type=float, default=1.0, help='Fraction of clients for evaluation')
    parser.add_argument('--perc', type=float, default=0.75, help='Percentage for label splitting')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 101112], help='List of seeds for experiments')
    parser.add_argument('--num_cpus', type=int, default=3, help='Number of CPUs per client')
    parser.add_argument('--num_gpus', type=float, default=0.0, help='Number of GPUs per client')
    parser.add_argument('--result_path', type=str, default='results/results_experiments_17.txt', help='Path to save the results')

    args = parser.parse_args()
    main(args)
