from typing import List, Tuple, cast
import numpy as np
import time
import os

# For the homomorphic encryption
import pickle
import tenseal as ts
from flwr.common import NDArray, NDArrays, Parameters
from functools import reduce


# /////////////////////// Homomorphic Encryption \\\\\\\\\\\\\\\\\\\\\\\\\\

def context():
    """
    This function is used to create the context of the homomorphic encryption:
    it is used to create the keys and the parameters of the encryption scheme (CKKS).

    :return: the context of the homomorphic encryption
    """
    cont = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        # This means that the coefficient modulus will contain 4 primes of 60 bits, 40 bits, 40 bits, and 60 bits.
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )

    cont.generate_galois_keys()  # You can create the Galois keys by calling generate_galois_keys
    cont.global_scale = 2 ** 40  # global_scale: the scaling factor, here set to 2**40 (same that pow(2, 40))
    return cont


class Layer:
    """
    This class is used to represent the weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: the weights of the layer
    """
    def __init__(self, name_layer, weight):
        self.name = name_layer
        self.weight_array = weight

    def get_name(self):
        """
        This function is used to get the name of the layer.
        """
        return self.name

    def get_weight(self):
        """
        This function is used to get the weights of the layer.
        """
        return self.weight_array

    def __add__(self, other):
        """
        This function is used to add the weights of two layers.

        :param other: the other layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array + weights)

    def __sub__(self, other):
        """
        This function is used to substract the weights of two layers.

        :param other: the other layer object

        :return: the substraction result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array - weights)

    def __mul__(self, other):
        """
        This function is used to multiply the weights of two layers.

        :param other: the other layer object

        :return: the multiplication result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array * weights)

    def __truediv__(self, other):
        """
        This function is used to divide the weights of two layers.
        It works only if the weights of the other layer are a number or a layer object (weights with the same shape).

        :param other: the other layer object

        :return: the division result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        weights = self.weight_array * (1 / weights)
        return Layer(self.name, weights)

    def __len__(self):
        """
        This function is used to get the number of weights of the layer (the size of the layer).
        """
        somme = 1
        for elem in self.weight_array.shape():
            somme *= elem
        return somme

    def shape(self):
        """
        This function is used to get the shape of the layer.
        """
        return self.weight_array.shape()

    def sum(self, axis=0):
        """
        This function is used to get the sum of the weights of the layer.

        :param axis: the axis of the sum (default: 0 for the sum of the columns)

        :return: the sum of the weights of the layer in the form of a layer object
        """
        return Layer(f"sum_{self.name}", self.weight_array.sum(axis=axis))

    def mean(self, axis=0):
        """
        This function is used to get the mean of the weights of the layer.

        :param axis: the axis of the mean (default: 0 for the mean of the columns)

        :return: the mean of the weights of the layer in the form of a layer object
        """
        weights = self.weight_array.sum(axis=axis) * (1 / self.weight_array.shape[axis])
        return Layer(f"sum_{self.name}", weights)

    def decrypt(self, sk=None):
        """
        This function is used to decrypt the weights of the layer.

        :param sk: the secret key used to decrypt the weights (default: None).
        Not used in this class but this is to respect the same structure as the CryptedLayer class.

        :return: the decrypted weights of the layer in the form of a list of weights
        """
        return self.weight_array.tolist()

    def serialize(self):
        """
        This function is used to serialize the weights of the layer.

        :return: the serialized weights of the layer in the form of a dictionary
        with the name of the layer as key and the weights as value
        """

        return {self.name: self.weight_array}


class CryptedLayer(Layer):
    """
    This class is used to represent the crypted weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: the crypted weights of the layer
    :param contexte: the context of the encryption
    """
    def __init__(self, name_layer, weight, contexte=None):
        super(CryptedLayer, self).__init__(name_layer, weight)
        if type(weight) == ts.tensors.CKKSTensor or type(weight) == bytes:
            # If the weights are already encrypted or if they are bytes (serialized weights)
            self.weight_array = weight

        else:
            # If the weights are not encrypted, we encrypt them with the context
            self.weight_array = ts.ckks_tensor(contexte, weight.cpu().detach().numpy())

    def __add__(self, other):
        """
        This function is used to add the crypted weights of two layers.

        :param other: the other layer object

        :return: the addition result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == CryptedLayer else other
        return CryptedLayer(self.name, self.weight_array + weights)

    def __sub__(self, other):
        """
        This function is used to substract the crypted weights of two layers.

        :param other: the other layer object

        :return: the substraction result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == CryptedLayer else other
        return CryptedLayer(self.name, self.weight_array - weights)

    def __mul__(self, other):
        """
        This function is used to multiply the crypted weights of two layers.

        :param other: the other layer object

        :return: the multiplication result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == CryptedLayer else other
        return CryptedLayer(self.name, self.weight_array * weights)

    def __truediv__(self, other):
        """
        This function is used to divide the crypted weights of two layers.

        :param other: the other layer object

        :return: the division result in the form of a crypted layer object
        """
        try:
            # We try to divide the weights of the layer by the weights of the other layer or by a number
            # It is possible only if the denominator is a number or a tensor of non-crypted weights
            weights = other.get_weight() if type(other) == CryptedLayer else other
            weights = self.weight_array * (1 / weights)

        except:
            print("Error: the division operator isn't supported by SEAL")
            weights = []

        return CryptedLayer(self.name, weights)

    def shape(self):
        """
        This function is used to get the shape of the layer.

        :return: the shape of the layer in the form of a tuple
        """
        return self.weight_array.shape

    def sum(self, axis=0):
        """
        This function is used to get the sum of the weights of the layer.

        :param axis: the axis of the sum (default: 0 for the sum of the columns)

        :return: the sum of the weights of the layer in the form of a crypted layer object
        """
        return CryptedLayer(f"sum_{self.name}", self.weight_array.sum(axis=axis))

    def mean(self, axis=0):
        """
        This function is used to get the mean of the weights of the layer.

        :param axis: the axis of the mean (default: 0 for the mean of the columns)

        :return: the average of the weights of the layer in the form of a crypted layer object
        """
        weights = self.weight_array.sum(axis=axis) * (1 / self.weight_array.shape[axis])
        return CryptedLayer(f"sum_{self.name}", weights)

    def decrypt(self, sk=None):
        """
        This function is used to decrypt the weights of the layer.

        :param sk: the secret key used to decrypt the weights (default: None)

        :return: the decrypted weights of the layer in the form of a list of weights
        """
        return self.weight_array.decrypt(sk).tolist() if sk else self.weight_array.decrypt().tolist()

    def serialize(self):
        """
        This function is used to serialize the weights of the layer.

        :return: the serialized weights of the layer in the form of a dictionary
        """
        return {self.name: self.weight_array.serialize()}


def crypte(client_w, context_c):
    """
    This function is used to crypte the weights of a neural network.

    :param client_w: dictionary of the weights of a neural network
    :param context_c: the context of the encryption
    :return: a list of Layer objects (crypted weights or not)
    """
    encrypted = []
    for name_layer, weight_array in client_w.items():
        start_time = time.time()
        if name_layer == 'fc3.weight':
            encrypted.append(CryptedLayer(name_layer, weight_array, context_c))

        else:
            encrypted.append(Layer(name_layer, weight_array))

        print(name_layer, (time.time() - start_time))

    # return [CryptedLayer(name_layer, weight_array, context_c) for name_layer, weight_array in client_w.items()]
    return encrypted


def read_query(file_path):
    """
    This function is used to read a pickle file.

    :param file_path: the path of the file to read
    :return: the query and the context
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            """
            # pickle.load(f)  # load to read file object

            file_str = f.read()
            client_query1 = pickle.loads(file_str)  # loads to read str class
            """
            query_str = pickle.load(file)

        contexte = query_str["contexte"]  # ts.context_from(query["contexte"])
        del query_str["contexte"]
        return query_str, contexte

    else:
        print("The file doesn't exist")


def write_query(file_path, client_query):
    """
    This function is used to write a pickle file.

    :param file_path: the path of the file to write

    :param client_query: the query to write
    """
    with open(file_path, 'wb') as file:  # 'ab' to add existing file
        encode_str = pickle.dumps(client_query)
        file.write(encode_str)


def deserialized_layer(name_layer, weight_array, ctx):
    """
    This function is used to deserialized a layer (crypted or not).
    i.e. to transform the weights of a layer in the correct format (not bytes).

    :param name_layer: the name of the layer
    :param weight_array: the weights of the layer
    :param ctx: the context (if the layer is crypted)
    :return: the object Layer or CryptedLayer with the weights of the layer in the correct format
    """
    if type(weight_array) == bytes:
        return CryptedLayer(name_layer, ts.ckks_tensor_from(ctx, weight_array), ctx)

    elif type(weight_array) == ts.tensors.CKKSTensor:
        return CryptedLayer(name_layer, weight_array, ctx)

    else:
        return Layer(name_layer, weight_array)


def deserialized_model(client_query, ctx):
    """
    This function is used to deserialized a model (crypted or not).
    i.e. to transform the weights of a model in the correct format (not bytes).

    :param client_query: the model
    :param ctx: the context (if the model is crypted)
    :return: the model with the weights in the correct format
    """
    return [deserialized_layer(name_layer, weight_array, ctx) for name_layer, weight_array in client_query.items()]


# Redefine the aggregate function (defined in Flower)
def aggregate_custom(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average.
    Args:
        results: List of tuples containing the model weights and the number of samples used to compute the weights.
    Returns: A list of model weights averaged according to the number of samples used to compute the weights.
    """
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) * (1/num_examples_total)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
