########
# Given .CSV file of all datasamples, split dataset in different client formations

# general imports
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import argparse
import random


# imports from libraries
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Subset, Dataset, DataLoader, random_split, ConcatDataset, StackDataset, ChainDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import check_all_classes


# general stuffies
current_path = os.getcwd()
all_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

central_df = ['Age', 'Class of Worker', 'Education', 'Marital Status', 'Occupation', 'Place of Birth', 'Worked hours', 'Sex', 'Race', 'State', 'Income']


data_folderpath = os.path.join(current_path, 'acs_data')
file_list = os.listdir(data_folderpath)
list_state_df = [pd.read_csv(os.path.join(data_folderpath, file)) for file in file_list]
df_all_states = pd.concat(list_state_df, ignore_index=True)


class CENSUSData(Dataset):
    def __init__(self, dataframe):
        # print('Dataframe is', dataframe)
        self.data = self.prepare_data(dataframe)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # getitem for RAW datase, get features and label
        feature_names = ['Sex', 'Race', 'Class of Worker', 'Education', 'Age', 'Marital Status', 'Occupation', 'Place of Birth', 'Worked hours']
        # feature_names = ['Sex']
        
        column_indices = [self.data.columns.get_loc(col) for col in feature_names]
        features = torch.tensor(self.data.iloc[idx, column_indices], dtype=torch.float32)
        label = torch.tensor(self.data.loc[idx, 'Income'], dtype=torch.long)

        return features, label
    
    def prepare_data(self, dataframe):
        '''Prepare data per individual .csv file of every state'''

        # load data
        # df_state = pd.read_csv(self.file_path)        
        # delete non-relevant columns
        feature_names = ['Sex', 'Race', 'Class of Worker', 'Education', 'Age', 'Marital Status', 'Occupation', 'Place of Birth', 'Worked hours', 'Income']
        
        data = dataframe[feature_names]
        
        # convert categorical features into numerical columns
        label_mappings = {
            'Income': {'<=50K': 0, '>50K': 1}
        }

        # Apply the mappings
        for feature, mapping in label_mappings.items():
            data[feature] = data[feature].map(mapping)

        # # Undersampling the majority class
        # count_class_0, count_class_1 = data['Income'].value_counts()
        # df_class_0 = data[data['Income'] == 0]
        # df_class_1 = data[data['Income'] == 1]
        
        # df_class_0_under = df_class_0.sample(count_class_1, random_state=42)  # Using a random state for reproducibility
        # data = pd.concat([df_class_0_under, df_class_1], axis=0)
        # data.reset_index(drop=True, inplace=True)
        # print(data['Income'])
        return data
    
def get_range(dataset, feature):
    return dataset[feature].unique()

def sample_by_feature(dataset, feature, value):
    """
    Sample all data points where the feature label equals value.
    Returns both the sampled data points as a list and the filtered DataFrame.
    """
    # print(f"I am sampling feature{feature} for value {value}")
    # Filter data where label equals value
    filtered_indices = dataset.index[dataset[feature] == value].tolist()
    
    # Fetch all the sampled data points
    sampled_data = dataset.iloc[filtered_indices]
    
    # Get the filtered DataFrame
    filtered_dataframe = dataset[dataset[feature] == value]
    
    return sampled_data, filtered_dataframe

def sample_by_feature_pairs(dataset, feature1, value1, feature2, value2):
    """
    Sample all data points where both feature1 and feature2 equal the given values.
    Returns both the sampled data points as a list and the filtered DataFrame.
    """
    # Filter data where both feature1 and feature2 equal the given values
    filtered_indices = dataset.index[(dataset[feature1] == value1) & (dataset[feature2] == value2)].tolist()
    
    # Fetch all the sampled data points
    sampled_data = [dataset[i] for i in filtered_indices]
    
    # Get the filtered DataFrame
    filtered_dataframe = dataset[(dataset[feature1] == value1) & (dataset[feature2] == value2)]
    
    return sampled_data, filtered_dataframe
    
def filepath_list(list_states):
    """list all file-paths for all states """
    file_list = []
    for state in list_states:
        file = 'ACS-2022' +'-' + str(state) + ".csv" 
        path = os.path.join(current_path, 'acs_data/', file)
        file_list.append(path)
    return file_list

def get_state_clients(list_states):
    clients = []
    data_folderpath = os.path.join(current_path, 'acs_data')
    for dataset_path in filepath_list(list_states):
        clientpath = os.path.join(data_folderpath, dataset_path)
        client_df = pd.read_csv(clientpath)
        dataset = CENSUSData(client_df)
        clients.append(dataset)
    return clients

def get_state_clients_synthetic():
    clients = []
    path = '/home/jelke/Documents/AI/Thesis/fl_code/synthetic_dataset.csv'
    dataset = pd.read_csv(path)
    grouped = dataset.groupby('State')
    for state_df in grouped:
        state_data = CENSUSData(state_df)
        clients.append(state_data)
    return clients

def load_data(clients, seed, batch_size, shuffle=True, missing=True):
    """Load tabular data from a CSV file and preprocess it ."""
    trainloaders = []
    valloaders = []
    testloaders = []
    missing_classes = {}

    print('-----------')
    print('Loading and preparing data...')
    # Load CSV data into a custom dataset    

    all_train, all_val, all_test = [], [], []

    for i, client_dataset in enumerate(clients):
        train_size = int(0.8 * len(client_dataset))
        val_size = int(0.1 * len(client_dataset))
        test_size = len(client_dataset) - (train_size + val_size)
        train_dataset, val_dataset, test_dataset = random_split(client_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

        trainloaders.append(DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=shuffle))
        valloaders.append(DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=False))
        testloaders.append(DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False))
        all_train.append(train_dataset)
        all_val.append(val_dataset)
        all_test.append(test_dataset)

        if missing:
            # print('Checking for missing classes....')
            train_missing, val_missing, test_missing = check_all_classes(train_dataset, val_dataset, test_dataset)
            if train_missing or val_missing or test_missing:
                missing_classes[i] = {'Train': train_missing, 'Val': val_missing, 'Test': test_missing}
    
    central_train_loader = DataLoader(ConcatDataset(all_train), batch_size=batch_size, drop_last=False, shuffle=True)
    central_val_loader = DataLoader(ConcatDataset(all_val), batch_size=batch_size, drop_last=False, shuffle=True)
    central_test_loader = DataLoader(ConcatDataset(all_test), batch_size=batch_size, drop_last=False, shuffle=True)
    
    print('Data is ready for use!')
    return trainloaders, valloaders, testloaders, central_train_loader, central_val_loader, central_test_loader, missing_classes

###########################################
# QUANTITY SPLITS
###########################################

def original_split():
    all_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    clients = get_state_clients(all_states)
    return clients

def split_quantity_binary(metadata, x):
    biggest_states = metadata.nlargest(x, 'Size')['State'].tolist()
    smallest_states = metadata.nsmallest(x, 'Size')['State'].tolist()
    list_states = biggest_states + smallest_states
    return get_state_clients(list_states)

def split_quantity_interval(metadata, x):
    interval = len(metadata) // (x*2)
    sorted_metadata = metadata.sort_values(by='Size')

    # Select states at regular intervals
    list_states = [sorted_metadata.iloc[i * interval]['State'] for i in range(x*2)]
    filepath_list(list_states)
    return get_state_clients(list_states)

def arange_client_sizes(min_val, max_val, num_values, target_sum):
    # Start with the original max_val and adjust until the sum meets the condition
    while True:
        values = np.linspace(min_val, max_val, num_values)
        rounded_values = np.round(values).astype(int)
        if sum(rounded_values) < target_sum:
            break
        else:
            max_val -= 1000  # Decrease max_val incrementally
    return rounded_values

def split_quantity_equal(num_clients, client_size, data_folderpath, seed):
    """
    BASELINE: non-state based --> sample datapoints"""
    df = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Calculate the size of each client dataset
    total_data_points = len(df_shuffled)
    client_size = total_data_points // 51

    clients = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size
        subset_df = df_shuffled.iloc[start_idx:end_idx]
        subset_df.reset_index(drop=True, inplace=True)
        clients.append(CENSUSData(subset_df))

    return clients

def split_quantity_nonequal(num_clients, client_sizes, data_folderpath, seed):
    """
    QUANTITY HETEROGENEITY: non-state based --> sample datapoints
    """
    df = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the DataFrame
    # print(len(df))

    assert num_clients == len(client_sizes), "Number of clients must match the length of client_sizes"

    # print('num clients', num_clients)
    # print(len(client_sizes))
    # print(client_sizes)
    clients = []
    start_idx = 0
    for size in client_sizes:
        end_idx = start_idx + size
        # print(end_idx)
        if end_idx > len(df_shuffled):
            # If the range exceeds the DataFrame length, truncate the range
            end_idx = len(df_shuffled)
        
        # Slice the DataFrame for the current client
        subset_df = df_shuffled.iloc[start_idx:end_idx]
        subset_df.reset_index(drop=True, inplace=True)
        
        # Assuming CENSUSData is a class or function that processes each client's DataFrame
        clients.append(CENSUSData(subset_df))

        # Update start_idx for the next client
        start_idx = end_idx

        # Break if we have allocated all available rows
        if start_idx >= len(df_shuffled):
            # print('done!')
            break
    # print(len(clients))
    return clients

def create_clients_equal(rough_clients, num_clients, seed):
    """
    Function that creates equally sized clients from rough_clients as given by data-split on label/feature

    Input: rough_clients, original data split based on label and/or feature
    Output: #num_clients equally sized clients that correspond to the label/feature split 
    """
    random.seed(seed)
    new_clients = []

    # print('number of rough_clients')
    
    # calculate how much every rough_client should be split in, should be at least 1
    new_client_per_rough = max(1, num_clients // len(rough_clients))
    # print('Number of new clients per rough client', new_client_per_rough)
    
    smallest_len = min(len(df) for df in rough_clients) // max(1, new_client_per_rough)
    # print('Small len', smallest_len)

    actual_client_counter = 0

    for i, r_client in enumerate(rough_clients):
        # print('Rough client number', i)
        r_client = r_client.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the DataFrame
        for i in range(new_client_per_rough):
            # print('number of client', actual_client_counter)
            sub_df = r_client.iloc[i * smallest_len: (i + 1) * smallest_len].reset_index(drop=True)
            # print(sub_df)
            # print('=======================================')
            new_clients.append(CENSUSData(sub_df))
            actual_client_counter += 1
        # print('-------------------------------------------------------------------')
    
    return new_clients

def split_sublists(lst, n):
    """Helper function that splits every sublist (DataFrame) into n smaller sublists (DataFrames)"""
    def split_dataframe(df, n):
        k, m = divmod(len(df), n)
        return [df.iloc[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
    
    result = []
    for df in lst:
        result.extend(split_dataframe(df, n))
    return result

def create_clients_unequal(split, num_clients, seed):
    """
    Function that creates equally sized clients from rough_clients as given by data-split on label/feature
    
    Input: split, original data split based on label and/or feature
    Output: #num_clients equally sized clients that correspond to the label/feature split 
    """

    # Calculate how many clients each split should be split into to obtain #num_clients clients
    new_split = num_clients // len(split)

    # Split the rough clients to create #num_clients new clients
    splitted_rough_clients = split_sublists(split, new_split)

    lens = [len(df) for df in splitted_rough_clients]

    # Pair each sublist with its corresponding length
    pairs = list(zip(splitted_rough_clients, lens))

    # The max subtraction possible such that no client is smaller than 320 samples
    len_difference = min(lens)

    # Create distributed list of subtractions that can be taken off the original sizes
    subs = np.round(np.linspace(1, len_difference, num_clients)).astype(int)
    new_clients = [CENSUSData(df.sample(n=length-sub, random_state=seed).reset_index(drop=True)) for (df, length), sub in zip(pairs, subs)]        

    return new_clients

def create_percentage_equal(split, num_clients, perc, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    # number of samples (per client)
    # Calculate proportion from both groups
    p1 = perc
    p2 = 1 - perc

    split_lengths = [len(df) for df in split]
    
    # Calculate max number of samples that can be extracted when perc% from the first group, (1- perc)% from the second group
    max_from_split_1_1 = int(split_lengths[0] // (num_clients * p1))
    max_from_split_2_1 = int(split_lengths[1] // (num_clients * p2))
    # check if perc% from first group is smaller than (1 - perc)% from second group
    n1 = min(max_from_split_1_1, max_from_split_2_1)
    
    # Calculate max number of samples that can be extracted when (1 - perc)% from the first group, perc% from the second group
    max_from_split_1_2 = int(split_lengths[1] // (num_clients * p1))
    max_from_split_2_2 = int(split_lengths[0] // (num_clients * p2))
    n2 = min(max_from_split_1_2, max_from_split_2_2)
    
    # Choose the scenario that gives the maximum number of samples per class
    samples_per_client, group_big = max((val, idx) for idx, val in enumerate([n1, n2]))
    total_groups = len(split)

    assert total_groups == 2, "Percentage split can only be done on binary features"

    available_data = [df.copy() for df in split]

    # reorder list such that perc group is first
    available_data_first =[available_data[group_big]] + available_data[:group_big] + available_data[group_big + 1:]

    clients = []

    for _ in range(num_clients):
        client_data = []
        remaining_size = samples_per_client
        
        while remaining_size > 0:
            
            # Randomly select two different groups from available data
            group_indices = random.sample(range(len(available_data_first)), 2)
            group1, group2 = available_data_first[group_indices[0]], available_data_first[group_indices[1]]
            
            # Calculate sizes for each group
            group1_size = int(remaining_size * p1)
            group2_size = remaining_size - group1_size
            
            # Ensure sizes do not exceed available samples
            group1_size = min(group1_size, len(group1))
            group2_size = min(group2_size, len(group2))

            print(group2_size)
            
            group1_sample = group1.sample(n=group1_size, random_state=seed)
            group2_sample = group2.sample(n=group2_size, random_state=seed)
            
            # Remove sampled data from available data
            available_data_first[group_indices[0]] = group1.drop(group1_sample.index)
            available_data_first[group_indices[1]] = group2.drop(group2_sample.index)
            
            client_data.append(group1_sample)
            client_data.append(group2_sample)
            
            remaining_size -= (group1_size + group2_size)
            
            # Remove empty dataframes from available_data
            available_data_first = [df for df in available_data_first if not df.empty]

        combined_client = pd.concat(client_data).sample(frac=1, random_state=seed).reset_index(drop=True)
        clients.append(CENSUSData(combined_client))
    
    return clients

def create_percentage_unequal(split, num_clients, perc, seed):
    
    # Total number of data samples
    split_lengths = [len(df) for df in split]
    total_samples = sum(split_lengths)

    # biggest class size that is possible for data sampling
    max_class = int(min(split_lengths)/num_clients * perc)
    
    client_sizes = arange_client_sizes(320, max_class, num_clients, total_samples)

     # Initialize clients list
    clients = []

    # Create copies of dataframes in split to keep track of unused samples
    available_data = [df.copy() for df in split]

    for client_size in client_sizes:
        client_data = []
        remaining_size = client_size
        
        while remaining_size > 0:
            if len(available_data) < 2:
                break  # Not enough groups to select from
            
            # Randomly select two different groups from available data
            group_indices = random.sample(range(len(available_data)), 2)
            group1, group2 = available_data[group_indices[0]], available_data[group_indices[1]]
            
            # Randomly decide percentages for group1 and group2 within limits
            group1_percentage = perc
            group2_percentage = 1 - group1_percentage
            
            # Calculate sizes for each group
            group1_size = int(remaining_size * group1_percentage)
            group2_size = remaining_size - group1_size
            
            # Ensure sizes do not exceed available samples
            group1_size = min(group1_size, len(group1))
            group2_size = min(group2_size, len(group2))
            
            group1_sample = group1.sample(n=group1_size, random_state=seed)
            group2_sample = group2.sample(n=group2_size, random_state=seed)
            
            # Remove sampled data from available data
            available_data[group_indices[0]] = group1.drop(group1_sample.index)
            available_data[group_indices[1]] = group2.drop(group2_sample.index)
            
            client_data.append(group1_sample)
            client_data.append(group2_sample)
            
            remaining_size -= (group1_size + group2_size)
            
            # Remove empty dataframes from available_data
            available_data = [df for df in available_data if not df.empty]

        combined_client = pd.concat(client_data).sample(frac=1, random_state=seed).reset_index(drop=True)
        clients.append(CENSUSData(combined_client))
    
    return clients


###########################################
# FEATURE/LABEL SPLITS
###########################################

def split_feature(feature, data_folderpath, num_clients, seed):
    """
    FEATURE HETEROGENEITY: sample based on feature, crop clients to equal quantities
    """
    split = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))

    # loop through all label categories
    for i in get_range(dataset, feature):

        # sample data points that have that category
        _, sampled_data_points = sample_by_feature(dataset, feature, i)
        split.append(sampled_data_points)

    # create equal clients from the rough divisions
    clients = create_clients_equal(split, num_clients, seed)

    return clients, len(clients)


def split_feature_percentage(feature, data_folderpath, num_clients, perc, seed):
    """
    FEATURE HETEROGENEITY: sample based on feature, crop clients to equal quantities
    """
    split = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))

    # loop through all label categories
    for i in get_range(dataset, feature):

        # sample data points that have that category
        _, sampled_data_points = sample_by_feature(dataset, feature, i)
        split.append(sampled_data_points)

    # create equal clients from the rough divisions
    clients = create_percentage_equal(split, num_clients, perc, seed)

    return clients, len(clients)

def split_label(data_folderpath, num_clients, seed):
    """
    LABEL HETEROGENEITY: sample based on label, crop data-split to equally sized clients
    """
    split = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))

    # loop through all label categories
    for i in get_range(dataset, 'Income'):
        # print('The current label class is', i)

        # sample data points that have that category
        _, sampled_data_points = sample_by_feature(dataset, 'Income', i)
        split.append(sampled_data_points)

    # create equal clients from the rough divisions
    clients = create_clients_equal(split, num_clients, seed)

    return clients, len(clients)

def split_label_percentage(data_folderpath, num_clients, perc, seed):
    """
    LABEL HETEROGENEITY: sample based on label, crop data-split to equally sized clients
    """
    split = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))

    # loop through all label categories
    for i in get_range(dataset, 'Income'):
        # print('The current label class is', i)

        # sample data points that have that category
        _, sampled_data_points = sample_by_feature(dataset, 'Income', i)
        split.append(sampled_data_points)

    # create equal clients from the rough divisions
    clients = create_percentage_equal(split, num_clients, perc, seed)

    return clients, len(clients)

def split_quantity_feature(feature, data_folderpath, num_clients, seed):
    """
    FEATURE + QUANTITY HETEROGENEITY: sample based on label, crop data-split to non-equally sized clients
    """
    split = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))

    # loop through all feature categories
    for i in get_range(dataset, feature):

        # sample data points that have that category
        _, sampled_data_points = sample_by_feature(dataset, feature, i)
        split.append(sampled_data_points)

    # create unequal clients from the rough divisions
    clients = create_clients_unequal(split, num_clients, seed)
    return clients, len(clients)


def split_quantity_feature_percentage(feature, data_folderpath, num_clients, perc, seed):
    """
    FEATURE + QUANTITY HETEROGENEITY: sample based on label, crop data-split to non-equally sized clients
    """
    split = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))

    # loop through all feature categories
    for i in get_range(dataset, feature):

        # sample data points that have that category
        _, sampled_data_points = sample_by_feature(dataset, feature, i)
        split.append(sampled_data_points)

    # create unequal clients from the rough divisions
    clients = create_percentage_unequal(split, num_clients, perc, seed)
    return clients, len(clients)


def split_quantity_label(data_folderpath, num_clients, seed):
    """
    FEATURE + QUANTITY HETEROGENEITY: sample based on label, crop data-split to non-equal sized clients
    """
    rough_clients = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))

    # loop through all label categories
    for i in get_range(dataset, 'Income'):

        # sample data points that have that category
        _, sampled_data_points = sample_by_feature(dataset, 'Income', i)
        rough_clients.append(sampled_data_points)

    # create unequal clients from the rough divisions
    clients = create_clients_unequal(rough_clients, num_clients, seed)

    return clients, len(clients)

def split_quantity_label_percentage(data_folderpath, num_clients, perc, seed):
    """
    FEATURE + QUANTITY HETEROGENEITY: sample based on label, crop data-split to non-equal sized clients
    """
    rough_clients = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))

    # loop through all label categories
    for i in get_range(dataset, 'Income'):

        # sample data points that have that category
        _, sampled_data_points = sample_by_feature(dataset, 'Income', i)
        rough_clients.append(sampled_data_points)

    # create unequal clients from the rough divisions
    clients = create_percentage_unequal(rough_clients, num_clients, perc, seed)

    return clients, len(clients)

def split_feature_label(feature, data_folderpath, num_clients, seed):
    """
    FEATURE + LABEL HETEROGENEITY: sample based on label and feature, crop data-split to equally sized clients
    """
    rough_clients = []

    # obtain all datapoints
    dataset = pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv'))
    
    # loop through all feature and label categories
    for i in get_range(dataset, feature):
        for j in get_range(dataset,'Income'):

            # sample data points that have that combination
            sampled_data_points = sample_by_feature_pairs(dataset, feature, i, 'Income', j)
            rough_clients.append(sampled_data_points)

    # create equal clients from the rough divisions
    clients = create_clients_equal(rough_clients, num_clients, seed)
    return clients, len(clients)

def split_quantity_feature_label(feature, data_folderpath, num_clients, seed):
    """
    FEATURE + LABEL + QUANTITY HETEROGENEITY: sample based on label and feature, crop data-split to non-equally sized clients
    """
    rough_clients = []

    # obtain all datapoints
    dataset = CENSUSData(pd.read_csv(os.path.join(data_folderpath, 'ACS-2022-complete.csv')))
    
    # loop through all feature and label categories
    for i in dataset.get_range(feature):
        for j in dataset.get_range('Income'):

            # sample data points that have that combination
            sampled_data_points = dataset.sample_by_feature_pairs(feature, i, 'Income', j)
            rough_clients.append(sampled_data_points)

    # create unequal clients from the rough divisions
    clients = create_clients_unequal(rough_clients, num_clients, seed)
    return clients, len(clients)

# def main():
#     num_clients = 4
#     seed = 42
#     clients, len_clients = split_feature_percentage("Sex", data_folderpath, num_clients, 0.75, seed)
#     # clients, len_clients = split_label_percentage(data_folderpath, num_clients, seed)
#     clients, len_clients = split_quantity_label_percentage(data_folderpath, num_clients, 0.75, seed)
#     for cl in clients:
#         print(len(cl))
#     trainloaders, valloaders, testloaders, central_train_loader, central_val_loader, central_test_loader, missing_classes = load_data(clients, seed, batch_size=32, shuffle=True, missing=True)
#     print(missing_classes)

# main()