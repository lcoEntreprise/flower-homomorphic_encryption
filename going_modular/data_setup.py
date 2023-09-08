"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .common import *

NUM_WORKERS = os.cpu_count()

# Normalization values for the different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'animaux': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'breast': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'histo': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    }


def split_data_client(dataset, num_clients, seed):
    """
    This function is used to split the dataset into train and test for each client.
    :param dataset: the dataset to split (type: torch.utils.data.Dataset)
    :param num_clients: the number of clients
    :param seed: the seed for the random split
    """
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * (num_clients - 1)
    lengths += [len(dataset) - sum(lengths)]
    ds = random_split(dataset, lengths, torch.Generator().manual_seed(seed))
    return ds


# Define model, architecture and dataset
# The DataLoaders downloads the training and test data that are then normalized.
def load_datasets(num_clients: int, batch_size: int, resize: int, seed: int, num_workers: int, splitter=10,
                  dataset="cifar", data_path="./data/", data_path_val=""):
    """
    This function is used to load the dataset and split it into train and test for each client.
    :param num_clients: the number of clients
    :param batch_size: the batch size
    :param resize: the size of the image after resizing (if None, no resizing)
    :param seed: the seed for the random split
    :param num_workers: the number of workers
    :param splitter: percentage of the training data to use for validation. Example: 10 means 10% of the training data
    :param dataset: the name of the dataset in the data folder
    :param data_path: the path of the data folder
    :param data_path_val: the absolute path of the validation data (if None, no validation data)
    :return: the train and test data loaders
    """

    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[dataset])]

    if dataset == "cifar":
        # Download and transform CIFAR-10 (train and test)
        transformer = transforms.Compose(
            list_transforms
        )
        trainset = datasets.CIFAR10(data_path + dataset, train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(data_path + dataset, train=False, download=True, transform=transformer)

    else:
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms

        transformer = transforms.Compose(list_transforms)
        supp_ds_store(data_path + dataset)
        supp_ds_store(data_path + dataset + "/train")
        supp_ds_store(data_path + dataset + "/test")
        trainset = datasets.ImageFolder(data_path + dataset + "/train", transform=transformer)
        testset = datasets.ImageFolder(data_path + dataset + "/test", transform=transformer)

    print(f"The training set is created for the classes : {trainset.classes}")

    # Split training set into `num_clients` partitions to simulate different local datasets
    datasets_train = split_data_client(trainset, num_clients, seed)
    if data_path_val:
        valset = datasets.ImageFolder(data_path_val, transform=transformer)
        datasets_val = split_data_client(valset, num_clients, seed)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        if data_path_val:
            # if we already have a validation dataset
            trainloaders.append(DataLoader(datasets_train[i], batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(datasets_val[i], batch_size=batch_size))

        else:
            len_val = int(len(datasets_train[i]) * splitter / 100)  # splitter % validation set
            len_train = len(datasets_train[i]) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(datasets_train[i], lengths, torch.Generator().manual_seed(seed))
            trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader
