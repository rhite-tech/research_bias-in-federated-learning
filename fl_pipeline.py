# general imports
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import warnings


# imports from libraries
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from logging import WARNING

# all flower helper functions 
from flwr.common import (FitRes, NDArrays, Parameters, Scalar, Metrics)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log

# imports from other files
from train_test import get_parameters, set_parameters, train, test, LogisticRegression, calculate_eo_ratio, calculate_dp_ratio, calculate_eo_difference, calculate_dp_difference

torch.backends.cudnn.benchmark = True

def run_fl(model, trainloaders, valloaders, complete_trainloader, complete_valloader, DEVICE, NUM_CLIENTS, NUM_ROUNDS, FRACTION_FIT, FRACTION_EVALUATE, LEARNING_RATE):

    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, cid, model, trainloader, valloader, central_valloader):
            self.cid = cid
            self.model = model
            self.trainloader = trainloader
            self.valloader = valloader
            self.central_valloader = central_valloader

        def get_parameters(self, config):
            print(f"[Client {self.cid}] get_parameters")
            parameters = get_parameters(self.model)
            return parameters

        def fit(self, parameters, config):
            print(f"[Client {self.cid}] fit, config: {config}")
            set_parameters(self.model, parameters)
            train(self.model, self.trainloader, epochs=1, lr=LEARNING_RATE)
            return get_parameters(self.model), len(self.trainloader), {}

        def evaluate(self, parameters, config):
            print(f"[Client {self.cid}] evaluate, config: {config}")
            set_parameters(self.model, parameters)
            loss, metrics = test(self.model, self.valloader)
            return float(loss), len(self.valloader), {"accuracy": float(metrics['accuracy']), "eo_sex":float(metrics['equalized_odds'][0]), "dp_sex":float(metrics['demographic_parity'][0]), "eo_race":float(metrics['equalized_odds'][1]), "dp_race":float(metrics['demographic_parity'][1]), "eo_white":float(metrics['equalized_odds'][2]), "dp_white":float(metrics['demographic_parity'][2]), "eo_black":float(metrics['equalized_odds'][3]), "dp_black":float(metrics['demographic_parity'][3])}
        
    def client_fn(cid) -> FlowerClient:
        print(f'len trainloaders = {len(trainloaders)}, client_id = {cid}')
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(cid, model, trainloader, valloader, complete_valloader).to_client()
    
    class SaveModelStrategy(fl.server.strategy.FedAvg, ):

        def initialize_parameters(
            self, client_manager: ClientManager
        ) -> Optional[Parameters]:
            """Initialize global model parameters."""
            model = LogisticRegression(9)
            ndarrays = get_parameters(model)
            return fl.common.ndarrays_to_parameters(ndarrays)
        
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate model weights using weighted average and store checkpoint"""

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)


                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(model.state_dict(), f"acs_trained_models/model_round_{server_round}.pth")

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            # central_metrics = loss, accuracy, eo, dp, _, _ = test(self.model, self.valloader, self.central_valloader, self.cid)

            return aggregated_parameters, metrics_aggregated
    
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        avg_eo_sex = [num_examples * m["eo_sex"] for num_examples, m in metrics]
        avg_dp_sex = [num_examples * m["dp_sex"] for num_examples, m in metrics]
        avg_eo_race = [num_examples * m["eo_race"] for num_examples, m in metrics]
        avg_dp_race = [num_examples * m["dp_race"] for num_examples, m in metrics]
        avg_eo_white = [num_examples * m["eo_white"] for num_examples, m in metrics]
        avg_dp_white = [num_examples * m["dp_white"] for num_examples, m in metrics]
        avg_eo_black = [num_examples * m["eo_black"] for num_examples, m in metrics]
        avg_dp_black = [num_examples * m["dp_black"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples), "dp sex": sum(avg_dp_sex)/sum(examples), "dp race": sum(avg_dp_race)/sum(examples), "dp white": sum(avg_dp_white)/sum(examples), "dp black": sum(avg_dp_black)/sum(examples), "eo sex": sum(avg_eo_sex) / sum(examples), "eo race": sum(avg_eo_race) / sum(examples), "eo white": sum(avg_eo_white) / sum(examples), "eo black": sum(avg_eo_black) / sum(examples)}


    def global_evaluate(model: LogisticRegression, central_valloader, NUM_ROUNDS):
        """Return an evaluation function for server-side evaluation."""

        def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

            loss_function = nn.CrossEntropyLoss()
            total_loss, correct_predictions, total_samples = 0, 0, 0.0
            model.eval()
            
            total_labels = []
            total_predicted = []
            total_sensitive_attributes_sex, total_sensitive_attributes_race = [], []

            # Evaluation loop
            model.eval()  # Set model to evaluation mode if it's a PyTorch model
            for i, batch in enumerate(central_valloader):

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

            # Create binary categorization of race-classes
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
                # if min class == white, reverse score
                dp_white = (1 - dp_white)

            # check that for dp black/non-black, the ratio is black (label 0) / non-black (label 1)
            if dp_min_class_black == 1:
                # if min class == white, reverse score
                dp_black = (1 - dp_black)

            # check that for dp white/non-white, the ratio is non-white (label 1) / white (label 0)
            if eo_min_class_white == 0:
                # if min class == white, reverse score
                eo_white = (1 - eo_white)

            # check that for dp black/non-black, the ratio is black (label 0) / non-black (label 1)
            if dp_min_class_black == 1:
                # if min class == white, reverse score
                eo_black = (1 - eo_black)
            
            # Calculate average loss and accuracy
            loss = total_loss / len(central_valloader)
            accuracy = correct_predictions / total_samples

            all_eo = [eo_sex, eo_race, eo_white, eo_black]
            all_dp = [dp_sex, dp_race, dp_white, dp_black]

            minmax_classes = {'DP': [{'min sex': dp_min_class_sex, 'max sex': dp_max_class_sex}, {'min race': dp_min_class_race, 'max race': dp_max_class_race}], 'EO':[{'min sex': eo_min_class_sex, 'max sex': eo_max_class_sex}, {'min race': eo_min_class_race, 'max race': eo_max_class_race}]}

            return loss, {"accuracy": accuracy, 'equalized_odds': all_eo, 'demographic_parity': all_dp, 'min_max_classes': minmax_classes}
        
        return evaluate

    client_resources = {"num_cpus": 3, "num_gpus": 0.0}
    if DEVICE.type == "cuda":
        # here we are assigning an entire GPU for each client.
        client_resources = {"num_cpus": 3, "num_gpus": 1.0}

    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    # Create FedAvg strategy with model saving and weighted average 
    strategy = SaveModelStrategy(
        fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
        fraction_evaluate=FRACTION_EVALUATE, # Sample 50% of available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        evaluate_fn = global_evaluate(model, complete_valloader, NUM_ROUNDS),
    )

    # Start simulation
    results_fl = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
    )

    # return federated learning results
    return results_fl