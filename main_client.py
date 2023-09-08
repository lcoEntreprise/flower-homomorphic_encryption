from client import *
from federated import *
"""
script to start the client with the federated learning pipeline with Flower.
"""

if __name__ == '__main__':
    print("start client")
    start_time = time.time()
    fl.client.start_numpy_client(server_address="[::]:8080",
                                 client=client_common(args.id_client, args.model_save, args.yaml_path, args.roc_path,
                                                      args.save_results, args.matrix_path,
                                                      args.batch_size, trainloaders, valloaders,
                                                      DEVICE, CLASSES, args.he, args.path_keys, args.path_crypted)
                                 )
    print(f"client Time = {time.time() - start_time} seconds")
