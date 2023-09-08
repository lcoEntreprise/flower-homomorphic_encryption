import torch.nn.functional
import flwr as fl
from going_modular import *

"""
Script to define the client side of the federated learning pipeline with Flower.
"""


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################
class FlowerClient(fl.client.NumPyClient):
    """
    The client class for the federated learning pipeline with Flower.
    This class is used to define the client-side logic for federated learning with Flower.
    It inherits from fl.client.NumPyClient. The only real difference between Client and NumPyClient is
    that NumPyClient takes care of serialization and deserialization for you.

    args:
        cid: client id (int)
        net: model (torch.nn.Module)
        trainloader: trainloader (torch.utils.data.DataLoader)
        valloader: valloader (torch.utils.data.DataLoader)
        device: device (torch.device)
        batch_size: batch size (int)
        save_results: path to save the results (str)
        matrix_path: path to save the confusion matrix (str)
        roc_path: path to save the roc curve (str)
        yaml_path: path to save the yaml file (str)
        he: boolean to use the homomorphic encryption (bool)
        classes: list of classes (list)
        context_client: context client (tenseal.context.Context)
    """

    def __init__(self, cid, net, trainloader, valloader, device, batch_size, save_results, matrix_path, roc_path,
                 yaml_path, he, classes, context_client):

        # Initialize the client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.device = device
        self.batch_size = batch_size
        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path
        self.yaml_path = yaml_path
        self.he = he
        self.classes = classes
        self.context_client = context_client

    def get_parameters(self, config):
        """
        get the model parameters and return them as a list of NumPy ndarray’s
        (which is what flwr.client.NumPyClient expects)

        args:
            config: config (dict)

        return:
            parameters: parameters (list)
        """
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters2(self.net, self.context_client)

    def fit(self, parameters, config):
        """
        - Update the parameters of the local model with the parameters received from the server
        - Train the updated model on the local train dataset (x_train/y_train)
        - Return the updated model parameters and the number of examples used for training to the server (for 1 given round)

        args:
            parameters: parameters (list)
            config: config (dict)

        return:
            parameters: parameters (list)
        """
        # Read values from config
        server_round = config['server_round']
        local_epochs = config['local_epochs']
        lr = float(config["learning_rate"])

        # Use values provided by the config
        print(f'[Client {self.cid}, round {server_round}] fit, config: {config}')

        # Update local model parameters
        set_parameters(self.net, parameters, self.context_client)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr,
                                    momentum=0.9)

        # Start training
        results = engine.train(self.net, self.trainloader, self.valloader, optimizer=optimizer, loss_fn=criterion,
                               epochs=local_epochs, device=self.device)

        # Save results
        if self.save_results:
            save_graphs(self.save_results, local_epochs, results, f"_Client {self.cid}")

        return get_parameters2(self.net, self.context_client), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """
        - Update the parameters of the local model with the parameters received from the server

        - Evaluate the updated model on the local test dataset (x_test/y_test)

        - Return the local loss and accuracy to the server

        args:
            parameters: parameters (list)
            config: config (dict)

        return:
            loss: loss (float)
            num_examples: number of examples (int)
            metrics: metrics (dict)
        """
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters, self.context_client)

        # Evaluate global model parameters on the local test data
        loss, accuracy, y_pred, y_true, y_proba = engine.test(self.net, self.valloader,
                                                              loss_fn=torch.nn.CrossEntropyLoss(), device=self.device)

        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_path:
                save_matrix(y_true, y_pred, self.save_results + self.matrix_path, self.classes)

            if self.roc_path:
                save_roc(y_true, y_proba, self.save_results + self.roc_path, len(self.classes))
        # Return results, including the custom accuracy metric
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# The client-side execution (training, evaluation) from the server-side
def client_common(cid: str,
                  model_save: str, path_yaml: str, path_roc: str, results_save: str, path_matrix: str,
                  batch_size: str, trainloaders, valloaders, DEVICE, CLASSES,
                  he=False, secret_path="", server_path=""):
    """
    args:
        cid: client id (str)
        model_save: path to save the model (str)
        path_yaml: path to save the yaml file (str)
        path_roc: path to save the roc curve (str)
        results_save: path to save the results (str)
        path_matrix: path to save the confusion matrix (str)
        batch_size: batch size (int)
        trainloaders: trainloaders (list)
        valloaders: valloaders (list)
        DEVICE: device (torch.device)
        CLASSES: classes (list)
        he: boolean to use the homomorphic encryption (bool)
        secret_path: path to save the secret key (str)
        server_path: path to save the server public key (str)

    return:
        FlowerClient: client (FlowerClient object)

    """
    # Load data
    # Note: each client gets a different trainloader/valloader, so each client will train and evaluate
    # on their own unique dataset.
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    context_client = None

    # Load model
    net = Net(num_classes=len(CLASSES)).to(DEVICE)

    # Homomorphic encryption
    if he:
        print("Run with homomorphic encryption")
        if os.path.exists(secret_path):
            # To get the existing public/private keys combination
            with open(secret_path, 'rb') as f:
                query = pickle.load(f)

            context_client = ts.context_from(query["contexte"])

        else:
            # To create the public/private keys combination
            context_client = security.context()
            with open(secret_path, 'wb') as f:  # 'ab' to add existing file
                encode = pickle.dumps({"contexte": context_client.serialize(save_secret_key=True)})
                f.write(encode)

        secret_key = context_client.secret_key()

    else:
        print("Run WITHOUT homomorphic encryption")

    # C) Update the local model with the parameters received from the server
    # to get the trained model and the trained parameters (optimizer, metrics, ...)
    if os.path.exists(model_save):
        print(" To get the checkpoint")
        checkpoint = torch.load(model_save, map_location=DEVICE)['model_state_dict']
        if he:
            print("to decrypt model")
            # To decrypt the parameters with the private key
            server_query, server_context = security.read_query(server_path)
            server_context = ts.context_from(server_context)
            for name in checkpoint:
                print(name)
                # To decrypt the parameters with the private key
                checkpoint[name] = torch.tensor(
                    security.deserialized_layer(name, server_query[name], server_context).decrypt(secret_key)
                )

        # Update network with the aggregated results
        net.load_state_dict(checkpoint)

    # Create a  single Flower client representing a single organization
    return FlowerClient(cid, net, trainloader, valloader, device=DEVICE, batch_size=batch_size,
                        matrix_path=path_matrix, roc_path=path_roc, save_results=results_save, yaml_path=path_yaml,
                        he=he, context_client=context_client, classes=CLASSES)
