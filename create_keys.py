import os
import tenseal as ts

from going_modular import security

"""
Script to create the public/private keys combination.
"""


def combo_keys(client_path="secret.pkl", server_path="server_key.pkl"):
    """
    To create the public/private keys combination
    args:
        client_path: path to save the secret key (str)
        server_path: path to save the server public key (str)
    """
    context_client = security.context()
    security.write_query(client_path, {"contexte": context_client.serialize(save_secret_key=True)})
    security.write_query(server_path, {"contexte": context_client.serialize()})

    _, context_client = security.read_query(client_path)
    _, context_server = security.read_query(server_path)

    context_client = ts.context_from(context_client)
    context_server = ts.context_from(context_server)
    print("Is the client context private?", ("Yes" if context_client.is_private() else "No"))
    print("Is the server context private?", ("Yes" if context_server.is_private() else "No"))


if __name__ == '__main__':
    secret_path = "secret.pkl"
    public_path = "server_key.pkl"
    if os.path.exists(secret_path):
        # To get the existing public/private keys combination
        print("it exists")
        _, context_client = security.read_query(secret_path)

    else:
        # To create the public/private keys combination
        combo_keys(client_path=secret_path, server_path=public_path)
