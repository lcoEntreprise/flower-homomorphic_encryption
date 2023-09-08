import argparse
import random
import torch
import shutil
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn
import pandas as pd
import torch.nn.functional
from collections import OrderedDict
from .security import *


def write_yaml(data, file_write='toyaml.yml', data1=None):
    """
    A function to write YAML file

    :param data: data to write in the YAML file
    :param file_write: path to save the YAML file
    :param data1: data to add in the YAML file (if we want to add data in the YAML file without overwriting it)
    :return: data (the data to write in the YAML file)
    """

    def accumul_time(time_key, data, data1):
        if time_key in data1 and time_key in data:
            data[time_key] += data1[time_key]

        return data

    path_yaml = os.path.join(*file_write.split('/')[:-1])
    print("save yaml in ", path_yaml)
    os.makedirs(path_yaml, exist_ok=True)
    with open(file_write, 'w') as f:
        if data1:
            data = accumul_time("Train_time", data, data1)
            data = accumul_time("Test_time", data, data1)
            data = {**data1, **data}

        yaml.dump(data, f)

    return data


def read_yaml(yaml_file='config.yml'):
    """
    A function to read YAML file

    :param yaml_file: path to the YAML file
    :return: config (the data in the YAML file)
    """

    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    return config


def choice_device(device):
    """
    A function to choose the device

    :param device: the device to choose (cpu, gpu or mps)
    """
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and device != "cpu":
        """
        on Mac : 
        - torch.backends.mps.is_available() ensures that the current MacOS version is at least 12.3+
        - torch.backends.mps.is_built() ensures that the current current PyTorch installation was built with MPS activated.
        """
        device = "mps"

    else:
        device = "cpu"

    return device


def classes_string(name_dataset):
    """
    A function to get the classes of the dataset

    :param name_dataset: the name of the dataset
    :return: classes (the classes of the dataset) in a tuple
    """
    if name_dataset == "cifar":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif name_dataset == "animaux":
        classes = ('cat', 'dog')

    elif name_dataset == "breast":
        classes = ('0', '1')

    elif name_dataset == "histo":
        classes = ('0', '1')

    else:
        print("Warning problem : unspecified dataset")
        return ()

    return classes


def parsing(description='PyTorch ImageNet Training'):
    """
    A function to work with command line arguments (argparse)

    :param description: the description of the model
    :return: argparser object (the arguments of the model)
    """
    # To define the argparse arguments
    # Create the top-level parser
    parent_parser = argparse.ArgumentParser(description="common", add_help=False)
    parent_parser.add_argument('--max_epochs', type=int, default=1)
    parent_parser.add_argument('--number_clients', type=int, default=2)
    parent_parser.add_argument('--length', type=int, default=None, help='size at the entrance of the model')
    parent_parser.add_argument('--batch_size', type=int, default=64)
    parent_parser.add_argument('--device', default='cpu', type=str,
                               help="- Choice of the device between cpu and gpu "
                                    "(cuda if compatible Nvidia and mps if on mac\n"
                                    "- The choice the output may be cpu even if you choose the gpu if the latter isn't "
                                    "compatible")
    parent_parser.add_argument('--dataset', default='cifar', help="choice of the dataset (cifar10 by default)")
    parent_parser.add_argument('--data_path', type=str, default='./data/', help='Path to the training data')
    parent_parser.add_argument('--data_path_val', type=str, default=None, help='Path to the validation dataset')
    parent_parser.add_argument('--model_save', type=str, default='', help='Path to save the central model')
    parent_parser.add_argument('--yaml_path', type=str, default='./results/results.yml',
                               help='Path to save the metrics results')
    parent_parser.add_argument('--seed', type=int, default=42)
    parent_parser.add_argument('--num_workers', type=int, default=0)
    parent_parser.add_argument('--split', default=10, type=int,
                               help='ratio (in percent) of the training dataset that will be used for the test '
                                    '(default : 10)')
    parent_parser.add_argument('--lr', default=0.001, type=float,
                               help='learning rate for the central model'
                                    '(default : 0.001)')
    parent_parser.add_argument('--he', default=False, required=False, action='store_true', dest="he",
                               help='True if we want to use the homomorphic encryption (by default : False)')
    parent_parser.add_argument('--path_keys', type=str, default="secret.pkl",
                               help='Path to get the combo private/public keys')
    parent_parser.add_argument('--path_public_key', type=str, default="server_key.pkl",
                               help='Path to get the the public key')
    parent_parser.add_argument('--path_crypted', type=str, default="server.pkl",
                               help='Path to save the crypted (and not crypted) weights')

    # Create the specific commands for the "classic ML"
    parser_ml = argparse.ArgumentParser(description="classic", add_help=False)
    parser_ml.add_argument('--matrix_path', type=str, default=None,
                           help='Path to save the confusion matrix')
    parser_ml.add_argument('--roc_path', type=str, default=None,
                           help='Path to save the roc figures')
    parser_ml.add_argument('--save_results', type=str, default=None,
                           help='Path to save the results')

    # Create the specific commands for the "server"
    parser_server = argparse.ArgumentParser(description="server", add_help=False)
    parser_server.add_argument('--frac_fit', type=float, default=1.0)
    parser_server.add_argument('--frac_eval', type=float, default=0.5)
    parser_server.add_argument('--min_fit_clients', type=int, default=2)
    parser_server.add_argument('--min_eval_clients', type=int, default=None)
    parser_server.add_argument('--min_avail_clients', type=int, default=2)

    parser_server.add_argument('--rounds', default=3, type=int,
                               help='number of rounds (default : 3)')

    # Create the specific commands for the "client"
    parser_client = argparse.ArgumentParser(description="client", add_help=False)
    parser_client.add_argument('--id_client', type=str, default=None,
                               help='client id (by default None)')

    # Create the final parser
    main_parser = argparse.ArgumentParser(description=description)

    # Create the subparser of the final parser
    service_subparsers = main_parser.add_subparsers(title="service",
                                                    dest="service_command")

    # Add specific command for each choice (classic, client, server and simulation)
    classic_subparser = service_subparsers.add_parser("run", help="classic ML",
                                                      parents=[parent_parser, parser_ml])

    # python client.py client
    client_subparser = service_subparsers.add_parser("client", help="client",
                                                     parents=[parent_parser, parser_ml, parser_client])

    # python server.py server
    server_subparser = service_subparsers.add_parser("server", help="server",
                                                     parents=[parent_parser, parser_server])

    simul_subparser = service_subparsers.add_parser("simulation", help="client",
                                                    parents=[parent_parser, parser_server, parser_ml, parser_client])

    return main_parser


# functions for the dataset creation
def supp_ds_store(path):
    """
    Delete the hidden file ".DS_Store" created on macOS

    :param path: path to the folder where the hidden file ".DS_Store" is
    """
    for i in os.listdir(path):
        if i == ".DS_Store":
            print("Deleting of the hidden file '.DS_Store'")
            os.remove(path + "/" + i)


def create_files_train_test(path_init, path_final, splitter):
    """
    Split the dataset from path_init into two datasets : train and test in path_final
    with the splitter ratio (in %). Example : if splitter = 10, 10% of the initial dataset will be in the test dataset.

    :param path_init: path of the initial dataset
    :param path_final: path of the final dataset
    :param splitter: ratio (in %) of the initial dataset that will be in the test dataset.
    """
    # Move a file from rep1 to rep2
    for classe in os.listdir(path_init):
        list_init = os.listdir(path_init + "/" + classe)
        size_test = int(len(list_init) * splitter/100)
        print("Before : ", len(list_init))
        for _ in range(size_test):
            e = random.choice(list_init)  # random choice of the path of an image
            list_init.remove(e)
            shutil.move(path_init + classe + "/" + e, path_final + classe + "/" + e)

        print("After", path_init + classe, ":", len(os.listdir(path_init + classe)))
        print(path_final + classe, ":", len(os.listdir(path_final + classe)))


def save_matrix(y_true, y_pred, path, classes):
    """
    Save the confusion matrix in the path given in argument.

    :param y_true: true labels (real labels)
    :param y_pred: predicted labels (labels predicted by the model)
    :param path: path to save the confusion matrix
    :param classes: list of the classes
    """
    # To get the confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # To normalize the confusion matrix
    cf_matrix_normalized = cf_matrix / np.sum(cf_matrix) * 10

    # To round up the values in the matrix
    cf_matrix_round = np.round(cf_matrix_normalized, 2)

    # To plot the matrix
    df_cm = pd.DataFrame(cf_matrix_round, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)

    plt.savefig(path)
    plt.close()


def save_roc(targets, y_proba, path, nbr_classes):
    """
    Save the roc curve in the path given in argument.

    :param targets: true labels (real labels)
    :param y_proba: predicted labels (labels predicted by the model)
    :param path: path to save the roc curve
    :param nbr_classes: number of classes
    """
    y_true = np.zeros(shape=(len(targets), nbr_classes))  # array-like of shape (n_samples, n_classes)
    for i in range(len(targets)):
        y_true[i, targets[i]] = 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nbr_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nbr_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nbr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nbr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    lw = 2
    for i in range(nbr_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label='Worst case')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC) Curve OvR")  # One vs Rest
    plt.legend(loc="lower right")  # loc="best"

    plt.savefig(path)
    plt.close()


def save_graphs(path_save, local_epoch, results, end_file=""):
    """
    Save the graphs in the path given in argument.

    :param path_save: path to save the graphs
    :param local_epoch: number of epochs
    :param results: results of the model (accuracy and loss)
    :param end_file: end of the name of the file
    """
    os.makedirs(path_save, exist_ok=True)  # to create folders results
    print("save graph in ", path_save)
    # plot training curves (train and validation)
    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc"], results["val_acc"]],
        "Epochs", "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=path_save + "Accuracy_curves" + end_file)

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss"], results["val_loss"]],
        "Epochs", "Loss",
        curve_labels=["Training loss", "Validation loss"], title="Loss curves",
        path=path_save + "Loss_curves" + end_file)


def plot_graph(list_xplot, list_yplot, x_label, y_label, curve_labels, title, path=None):
    """
    Plot the graph of the list of points (list_xplot, list_yplot)
    :param list_xplot: list of list of points to plot (one line per curve)
    :param list_yplot: list of list of points to plot (one line per curve)
    :param x_label: label of the x axis
    :param y_label: label of the y axis
    :param curve_labels: list of labels of the curves (curve names)
    :param title: title of the graph
    :param path: path to save the graph
    """
    lw = 2

    plt.figure()
    for i in range(len(curve_labels)):
        plt.plot(list_xplot[i], list_yplot[i], lw=lw, label=curve_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if curve_labels:
        plt.legend(loc="lower right")

    if path:
        plt.savefig(path)


def get_parameters2(net, context_client=None) -> List[np.ndarray]:
    """
    Get the parameters of the network
    :param net: network to get the parameters (weights and biases)
    :param context_client: context of the crypted weights (if None, return the clear weights)
    :return: list of parameters (weights and biases) of the network
    """
    if context_client:
        # Crypte of the model trained at the client for a given round (after each round the model is aggregated between
        # clients)
        encrypted_tensor = crypte(net.state_dict(), context_client)  # list of encrypted layers (weights and biases)

        return [layer.get_weight() for layer in encrypted_tensor]

    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray], context_client=None):
    """
    Update the parameters of the network with the given parameters (weights and biases)
    :param net: network to set the parameters (weights and biases)
    :param parameters: list of parameters (weights and biases) to set
    :param context_client: context of the crypted weights (if None, set the clear weights)
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    if context_client:
        secret_key = context_client.secret_key()
        dico = {k: deserialized_layer(k, v, context_client) for k, v in params_dict}

        state_dict = OrderedDict(
            {k: torch.Tensor(v.decrypt(secret_key)) for k, v in dico.items()}
        )

    else:
        dico = {k: torch.Tensor(v) for k, v in params_dict}
        state_dict = OrderedDict(dico)

    net.load_state_dict(state_dict, strict=True)
    print("Updated model")
