from client import *
from server import *

"""
Script to start the simulation of the federated learning pipeline with Flower (client and server).
"""


# Implementing a Flower client
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    return client_common(cid,
                         model_save, path_yaml, path_roc, results_save, path_matrix,
                         batch_size, trainloaders, valloaders, DEVICE, CLASSES, he, secret_path, server_path)


# ////////////////////////////// Simulation of the federated learning pipeline with Flower ////////////////////////////
# Pass parameters to the Strategy for server-side parameter initialization
if __name__ == '__main__':
    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    """
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}
    """
    model_save = args.model_save
    path_yaml = args.yaml_path
    path_roc = args.roc_path
    results_save = args.save_results
    path_matrix = args.matrix_path
    batch_size = args.batch_size
    he = args.he
    secret_path = args.path_keys
    server_path = args.path_crypted

    print("Start simulation")
    start_simulation = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.number_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        # client_resources=client_resources
    )
    print(f"Simulation Time = {time.time() - start_simulation} seconds")
