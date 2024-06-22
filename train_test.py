# general imports
import numpy as np
import os
import torch
import torch.nn as nn

# imports from libraries
from collections import OrderedDict
from typing import List

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_path = os.getcwd()

#################################################
# Different classification models
#################################################

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 2)

    def forward(self, x):
        return self.linear(x)
    
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, num_features):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(num_features, 50)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(50, 2)  # Assuming binary classification
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

#################################################
# Train and test pipeline
#################################################  
  
def train(model, trainloader, epochs: int, lr, verbose=True):
    """Train model on the training set."""

    print('-----------')
    print('Training the model...')

    # define loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    train_samples = []

    # loop through epoch
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        # loop through batches
        for batch in trainloader:
            features, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # optimize parameters 
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate loss and accuracy 
            epoch_loss += loss.item()
            total += labels.size(0)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == labels).sum().item()
        # print('#correct, #total, epoch loss after training, in epoch', correct, total, epoch_loss)

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        print('-'*20)

    return train_samples


def test(model, valloader):
    """Evaluate model on the entire test set."""

    print('-----------')
    print('Testing the model...')

    loss_function = nn.CrossEntropyLoss()
    total_loss, correct_predictions, total_samples = 0, 0, 0.0

    # correct_predictions_male, correct_predictions_female = 0, 0
    model.eval()
    
    total_labels = []
    total_predicted = []
    total_sensitive_attributes_sex, total_sensitive_attributes_race = [], []

    with torch.no_grad():
        for i, batch in enumerate(valloader):

            features, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # Forward pass
            outputs = model(features)

            # Update metrics
            total_loss += loss_function(outputs, labels).item()
            _, preds = torch.max(outputs.data, 1)  # Convert probabilities to binary predictions (0 or 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # get list of sensitive attributes from batch
            sensitive_attributes_sex = features[:, 0]
            sensitive_attributes_race = features[:, 1]

            # prepare labels, predictions and sensitive features for fairlearn
            total_labels += labels.cpu()
            total_predicted += preds.cpu()
            total_sensitive_attributes_sex += sensitive_attributes_sex.cpu()
            total_sensitive_attributes_race += sensitive_attributes_race.cpu()

        # Convert to binary race-categorizations
        white_nonwhite = [0 if x == 1 else 1 for x in total_sensitive_attributes_race]
        black_nonblack = [0 if x == 2 else 1 for x in total_sensitive_attributes_race]

        # obtain fairness metrics for both sensitive attributes
        dp_sex, dp_min_class_sex, dp_max_class_sex = calculate_dp_ratio(torch.Tensor(total_labels), torch.Tensor(total_predicted), torch.Tensor(total_sensitive_attributes_sex))
        dp_race, dp_min_class_race, dp_max_class_race = calculate_dp_ratio(torch.Tensor(total_labels), torch.Tensor(total_predicted), torch.Tensor(total_sensitive_attributes_race))
        dp_white, dp_min_class_white, dp_max_class_white = calculate_dp_ratio(torch.Tensor(total_labels), torch.Tensor(total_predicted), torch.Tensor(white_nonwhite))
        dp_black, dp_min_class_black, dp_max_class_black = calculate_dp_ratio(torch.Tensor(total_labels), torch.Tensor(total_predicted), torch.Tensor(black_nonblack))
        eo_sex, eo_min_class_sex, eo_max_class_sex = calculate_eo_ratio(torch.Tensor(total_labels), torch.Tensor(total_predicted), torch.Tensor(total_sensitive_attributes_sex))
        eo_race, eo_min_class_race, eo_max_class_race = calculate_eo_ratio(torch.Tensor(total_labels), torch.Tensor(total_predicted), torch.Tensor(total_sensitive_attributes_race))
        eo_white, eo_min_class_white, eo_max_class_white = calculate_eo_ratio(torch.Tensor(total_labels), torch.Tensor(total_predicted), torch.Tensor(white_nonwhite))
        eo_black, eo_min_class_black, eo_max_class_black = calculate_eo_ratio(torch.Tensor(total_labels), torch.Tensor(total_predicted), torch.Tensor(black_nonblack))

        # check that for dp white/non-white, the ratio is non-white (label 1) / white (label 0)
        if dp_min_class_white == 0:
            print('dp min class = white')
            # if min class == white, reverse score
            dp_white = (1 - dp_white)

        # check that for dp black/non-black, the ratio is black (label 0) / non-black (label 1)
        if dp_min_class_black == 1:
            print('dp min class == black')
            # if min class == white, reverse score
            dp_black = (1 - dp_black)

        # check that for dp white/non-white, the ratio is non-white (label 1) / white (label 0)
        if eo_min_class_white == 0:
            print('eo min class = white')
            # if min class == white, reverse score
            eo_white = (1 - eo_white)

        # check that for dp black/non-black, the ratio is black (label 0) / non-black (label 1)
        if eo_min_class_black == 1:
            print('eo min class = black')
            # if min class == white, reverse score
            eo_black = (1 - eo_black)

        # obtain all minimal and maximal class-labels
        minmax_classes = {'DP': [{'min sex': dp_min_class_sex, 'max sex': dp_max_class_sex}, {'min race': dp_min_class_race, 'max race': dp_max_class_race}], 'EO':[{'min sex': eo_min_class_sex, 'max sex': eo_max_class_sex}, {'min race': eo_min_class_race, 'max race': eo_max_class_race}]}

    # Calculate average loss and accuracy
    loss = total_loss / len(valloader)
    accuracy = correct_predictions / total_samples

    all_eo = [eo_sex, eo_race, eo_white, eo_black]
    all_dp = [dp_sex, dp_race, dp_white, dp_black]

    return loss, {"accuracy": accuracy, 'equalized_odds': all_eo, 'demographic_parity': all_dp, 'minmax_classes': minmax_classes}, 


def calculate_eo_ratio(labels, preds, sensitive_attributes):
    """Calculate equalized odds ratio score based on predictions, labels and sensitive attribute labels"""
            
    correct_predictions_per_class = {}
    true_positives_per_class = {}
    false_positives_per_class = {}
    total_positives_per_class = {}
    total_negatives_per_class = {}
    # Initialize dictionary entries for new classes
    for attr in torch.unique(sensitive_attributes):
        if attr.item() not in correct_predictions_per_class:
            correct_predictions_per_class[attr.item()] = 0
            true_positives_per_class[attr.item()] = 0
            false_positives_per_class[attr.item()] = 0
            total_positives_per_class[attr.item()] = 0
            total_negatives_per_class[attr.item()] = 0

    # Calculate correct predictions per class
    for attr in torch.unique(sensitive_attributes):
        attr_mask = (sensitive_attributes == attr)
        correct_predictions_per_class[attr.item()] += ((preds == labels) & attr_mask).sum().item()

    # Calculate TPR and FPR components
    for label, pred, attr in zip(labels, preds, sensitive_attributes):
        if label == 1:
            total_positives_per_class[attr.item()] += 1
            if pred == 1:
                true_positives_per_class[attr.item()] += 1
        else:
            total_negatives_per_class[attr.item()] += 1
            if pred == 1:
                false_positives_per_class[attr.item()] += 1

    # Calculate TPR and FPR for each class
    tpr_per_class = {}
    fpr_per_class = {}
    for attr in true_positives_per_class:
        tpr_per_class[attr] = true_positives_per_class[attr] / total_positives_per_class[attr] if total_positives_per_class[attr] != 0 else np.nan
        fpr_per_class[attr] = false_positives_per_class[attr] / total_negatives_per_class[attr] if total_negatives_per_class[attr] != 0 else np.nan

    # Calculate min and max TPR and FPR
    min_tpr_class, min_tpr = min(tpr_per_class.items(), key=lambda x: x[1])
    max_tpr_class, max_tpr = max(tpr_per_class.items(), key=lambda x: x[1])
    min_fpr_class, min_fpr = min(fpr_per_class.items(), key=lambda x: x[1])
    max_fpr_class, max_fpr = max(fpr_per_class.items(), key=lambda x: x[1])

    # Calculate EO_ratio and determine which value (TPR or FPR) contributes to it
    tpr_ratio = min_tpr / max_tpr if max_tpr != 0 else np.nan
    fpr_ratio = min_fpr / max_fpr if max_fpr != 0 else np.nan

    if tpr_ratio < fpr_ratio:
        EO_ratio = tpr_ratio
        min_key, max_key = min_tpr_class, max_tpr_class
    else:
        EO_ratio = fpr_ratio
        min_key, max_key = min_fpr_class, max_fpr_class

    return EO_ratio, min_key, max_key


def calculate_eo_difference(labels, preds, sensitive_attributes):
    """Calculate equalized odds difference based on predictions, labels and sensitive attribute labels"""

    true_positives_per_class = {}
    false_positives_per_class = {}
    total_positives_per_class = {}
    total_negatives_per_class = {}

    # Initialize dictionary entries for new classes
    for attr in torch.unique(sensitive_attributes):
        true_positives_per_class[attr.item()] = 0
        false_positives_per_class[attr.item()] = 0
        total_positives_per_class[attr.item()] = 0
        total_negatives_per_class[attr.item()] = 0

    # Calculate TPR and FPR components
    for label, pred, attr in zip(labels, preds, sensitive_attributes):
        if label == 1:
            total_positives_per_class[attr.item()] += 1
            if pred == 1:
                true_positives_per_class[attr.item()] += 1
        else:
            total_negatives_per_class[attr.item()] += 1
            if pred == 1:
                false_positives_per_class[attr.item()] += 1

    # Calculate TPR and FPR for each class
    tpr_per_class = {}
    fpr_per_class = {}
    for attr in true_positives_per_class:
        tpr_per_class[attr] = true_positives_per_class[attr] / total_positives_per_class[attr] if total_positives_per_class[attr] != 0 else np.nan
        fpr_per_class[attr] = false_positives_per_class[attr] / total_negatives_per_class[attr] if total_negatives_per_class[attr] != 0 else np.nan

    # Calculate min and max TPR and FPR
    min_tpr_class, min_tpr = min(tpr_per_class.items(), key=lambda x: x[1])
    max_tpr_class, max_tpr = max(tpr_per_class.items(), key=lambda x: x[1])
    min_fpr_class, min_fpr = min(fpr_per_class.items(), key=lambda x: x[1])
    max_fpr_class, max_fpr = max(fpr_per_class.items(), key=lambda x: x[1])

    # Calculate EO_difference
    tpr_diff = abs(max_tpr - min_tpr)
    fpr_diff = abs(max_fpr - min_fpr)
    EO_diff = max(tpr_diff, fpr_diff)

    return EO_diff, (min_tpr_class if tpr_diff >= fpr_diff else min_fpr_class), (max_tpr_class if tpr_diff >= fpr_diff else max_fpr_class)


def calculate_dp_ratio(labels, preds, sensitive_attributes):
    """Calculate demographic parity ratio score based on predictions, labels and sensitive attribute labels"""
    
    positives_per_class = {}
    total_per_class = {}

    # print('Sens_attributes', sensitive_attributes)

    # Initialize dictionary entries for new classes
    for attr in torch.unique(sensitive_attributes):
        # print('Attr', attr)
        if attr.item() not in positives_per_class:
            positives_per_class[attr.item()] = 0
            total_per_class[attr.item()] = 0

    # Calculate the number of positive predictions per class
    for label, pred, attr in zip(labels, preds, sensitive_attributes):
        attr_item = attr.item()
        if pred == 1:
            positives_per_class[attr_item] += 1
        total_per_class[attr_item] += 1

    # Calculate the expected positive rate for each class
    expected_positive_rate_per_class = {}
    for attr in positives_per_class:
        expected_positive_rate_per_class[attr] = positives_per_class[attr] / total_per_class[attr] if total_per_class[attr] != 0 else np.nan

    # Calculate min and max expected positive rates, store the class
    min_positive_rate_class, min_positive_rate = min(expected_positive_rate_per_class.items(), key=lambda x: x[1])
    max_positive_rate_class, max_positive_rate = max(expected_positive_rate_per_class.items(), key=lambda x: x[1])

    # Calculate DP_ratio
    DP_ratio = min_positive_rate / max_positive_rate if max_positive_rate != 0 else np.nan

    return DP_ratio, min_positive_rate_class, max_positive_rate_class


def calculate_dp_difference(labels, preds, sensitive_attributes):
    """Calculate demographic parity difference based on predictions, labels and sensitive attribute labels"""

    positives_per_class = {}
    total_per_class = {}

    # Initialize dictionary entries for new classes
    for attr in torch.unique(sensitive_attributes):
        positives_per_class[attr.item()] = 0
        total_per_class[attr.item()] = 0

    # Calculate the number of positive predictions per class
    for pred, attr in zip(preds, sensitive_attributes):
        attr_item = attr.item()
        if pred == 1:
            positives_per_class[attr_item] += 1
        total_per_class[attr_item] += 1

    # Calculate the expected positive rate for each class
    expected_positive_rate_per_class = {}
    for attr in positives_per_class:
        expected_positive_rate_per_class[attr] = positives_per_class[attr] / total_per_class[attr] if total_per_class[attr] != 0 else np.nan

    # Calculate min and max expected positive rates
    min_positive_rate_class, min_positive_rate = min(expected_positive_rate_per_class.items(), key=lambda x: x[1])
    max_positive_rate_class, max_positive_rate = max(expected_positive_rate_per_class.items(), key=lambda x: x[1])

    # Calculate DP_difference
    DP_diff = abs(max_positive_rate - min_positive_rate)

    return DP_diff, min_positive_rate_class, max_positive_rate_class


def get_parameters(net) -> List[np.ndarray]:
    parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return parameters


def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
