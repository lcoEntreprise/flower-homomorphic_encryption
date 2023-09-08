import torch.nn.functional
from typing import Optional, Callable, Union
from logging import WARNING
import flwr.server.strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import (
    Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, Scalar, logger,
    ndarrays_to_parameters_custom, parameters_to_ndarrays_custom
)

from federated import *
from going_modular.security import aggregate_custom

"""
Script with the server-side logic for federated learning with Flower.
"""
# #############################################################################
# 3. Federated learning strategy
# #############################################################################
central = Net(num_classes=len(CLASSES)).to(DEVICE)


# Weighted averaging function to aggregate the accuracy metric we return from evaluate
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Function will be by Flower called after every round : Centralized Evaluation (or server-side evaluation)
def evaluate2(server_round: int, parameters: NDArrays,
              config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    set_parameters(central, parameters)  # Update model with the latest parameters
    loss, accuracy, y_pred, y_true, y_proba = engine.test(central, testloader, loss_fn=torch.nn.CrossEntropyLoss(),
                                                          device=DEVICE)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def get_on_fit_config_fn(epoch=2, lr=0.001, batch_size=32) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, str]:
        """
        Return training configuration dict for each round with static batch size and (local) epochs.

        Perform two rounds of training with one local epoch, increase to two local epochs afterwards.
        """
        config = {
            "learning_rate": str(lr),
            "batch_size": str(batch_size),
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": epoch  # 1 if server_round < 2 else epoch,
        }
        return config

    return fit_config


def aggreg_fit_checkpoint(server_round, aggregated_parameters, central_model, path_checkpoint,
                          context_client=None, server_path=""):
    if aggregated_parameters is not None:
        print(f"Saving round {server_round} aggregated_parameters...")

        # Convert `Parameters` to `List[np.ndarray]`
        aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays_custom(aggregated_parameters, context_client)
        """
        [value of 'conv1.weight', 
        value of 'conv1.bias', 
        value of 'conv2.weight', 
        value of 'conv2.bias', 
        value of 'fc1.weight', 
        value of 'fc1.bias', 
        value of 'fc2.weight', 
        value of 'fc2.bias', 
        value of 'fc3.weight', 
        value of 'fc3.bias']
        """
        if context_client:
            def serialized(key, matrix):
                if key == 'fc3.weight':
                    return matrix.serialize()
                else:
                    return matrix

            server_response = {
                **{
                    key: serialized(key, aggregated_ndarrays[i]) for i, key in enumerate(central_model.state_dict().keys())},  # bytes
                "contexte": server_context.serialize()
            }

            security.write_query(server_path, server_response)

        else:
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(central_model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            central_model.load_state_dict(state_dict, strict=True)

            # Save the model
            if path_checkpoint:
                torch.save({
                    'model_state_dict': central_model.state_dict(),
                }, path_checkpoint)


# A Strategy from scratch with the same sampling of the clients as it is in FedAvg
# and then change the configuration dictionary
class FedCustom(fl.server.strategy.Strategy):
    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                    Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
                ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            context_client=None
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn,
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.context_client = context_client

    def __repr__(self) -> str:
        # Same function as FedAvg(Strategy)
        return f"FedCustom (accept_failures={self.accept_failures})"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # Same function as FedAvg(Strategy)
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        # Same function as FedAvg(Strategy)
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        # Custom fit config function provided
        standard_lr = args.lr
        higher_lr = 0.003
        config = {"server_round": server_round, "local_epochs": 1}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # fit_ins = FitIns(parameters, config)
        # Return client/config pairs
        fit_configurations = []
        for idx, client in enumerate(clients):
            config["learning_rate"] = standard_lr if idx < half_clients else higher_lr
            """
            Each pair of (ClientProxy, FitRes) constitutes 
            a successful update from one of the previously selected clients.
            """
            fit_configurations.append(
                (
                    client,
                    FitIns(
                        parameters,
                        config
                    )
                )
            )
        # Successful updates from the previously selected and configured clients
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average. (each round)"""
        # Same function as FedAvg(Strategy)
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results parameters --> array matrix
        weights_results = [
            (parameters_to_ndarrays_custom(fit_res.parameters, self.context_client), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Aggregate parameters using weighted average between the clients and convert back to parameters object (bytes)
        parameters_aggregated = ndarrays_to_parameters_custom(aggregate_custom(weights_results))

        metrics_aggregated = {}
        # Aggregate custom metrics if aggregation fn was provided
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Same function as SaveModelStrategy(fl.server.strategy.FedAvg)
        """Aggregate model weights using weighted average and store checkpoint"""
        aggreg_fit_checkpoint(server_round, parameters_aggregated, central, args.model_save,
                              self.context_client, args.path_crypted)
        return parameters_aggregated, metrics_aggregated

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        # Same function as FedAvg(Strategy)
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Same function as FedAvg(Strategy)
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}  # {"server_round": server_round, "local_epochs": 1}

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        # Each pair of (ClientProxy, FitRes) constitutes a successful update from one of the previously selected clients
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        # Same function as FedAvg(Strategy)
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {}
        # Aggregate custom metrics if aggregation fn was provided
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        # Only log this warning once
        elif server_round == 1:
            logger.log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # Same function as FedAvg(Strategy)
        if self.evaluate_fn is None:
            # Let's assume we won't perform the global model evaluation on the server side.
            return None

        # if we have a global model evaluation on the server side :
        parameters_ndarrays = parameters_to_ndarrays_custom(parameters, self.context_client)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        # if you haven't results
        if eval_res is None:
            return None

        loss, metrics = eval_res
        return loss, metrics


"""
- Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. 

- `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
if args.he:
    print("get public key : ", args.path_public_key)
    _, server_context = security.read_query(args.path_public_key)
    server_context = ts.context_from(server_context)

else:
    print("not HE so not key")
    server_context = None

strategy = FedCustom(
    fraction_fit=args.frac_fit,  # Train on frac_fit % clients (each round)
    fraction_evaluate=args.frac_eval,  # Sample frac_eval % of available clients for evaluation
    min_fit_clients=args.min_fit_clients,  # Never sample less than 10 clients for training
    min_evaluate_clients=args.min_eval_clients if args.min_eval_clients else args.number_clients // 2,
    min_available_clients=args.min_avail_clients,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    initial_parameters=ndarrays_to_parameters_custom(get_parameters2(central)),  # prevents Flower from asking one of the clients for the initial parameters
    evaluate_fn=None if args.he else evaluate2,  # Pass the evaluation function for the server side
    on_fit_config_fn=get_on_fit_config_fn(epoch=args.max_epochs, lr=args.lr, batch_size=args.batch_size),  # Pass the fit_config function
    context_client=server_context,
)

