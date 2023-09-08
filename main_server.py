from server import *
"""
Script to start the server side of the federated learning pipeline with Flower.
"""

if __name__ == '__main__':
    print("start server")
    start_time = time.time()

    fl.server.start_server(server_address="0.0.0.0:8080",
                           config=fl.server.ServerConfig(num_rounds=args.rounds),
                           strategy=strategy)
    print(f"Server Time = {time.time() - start_time} seconds")
