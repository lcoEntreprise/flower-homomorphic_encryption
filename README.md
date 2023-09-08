# Secure FL applied to medical imaging with fully homomorphic encryption
This GitHub contains the code used to run the experiments presented in a scientific paper (to be published very soon) and a [presentation given at Flower Summit 2023](https://youtu.be/pAvex7tpq2w?si=_sOmVMjiyA3cI0E5).
## Configure an environment
1. Creating an environment
    ```
    conda create -n fl_env python=3.10 anaconda
    ```
2. Add the necessary libraries with pip
   - If you want to install a specific version of Flower :
      - You can use this command (example with version 1.4.0) : : `pip install flower==1.4.0` 
      - You have to change `parameters.py` and `__init__.py` in flower library from this folder : flower/src/py/flwr/common (see modification in this github : https://github.com/data-science-lover/flower.git)

   ```
   pip install -r requirements.txt
   pip install pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   git clone https://github.com/data-science-lover/flower.git
   cd flower
   pip install .
   ```
3. Activate the environment
   ```
   conda activate fl_env
   ```
4. Add a specific dataset
   - If folder where put the datasets (for example called `data`) doesn't exist : create this folder and subfolder with the name of the specific dataset (example `cifar`): `./data/cifar`
   - In data, you have all datasets and in a specific dataset you have train and test folder: `./data/cifar/train/` 
   - If you have a specific dataset for the validation then addi the path to the folder with this command : `--data_path_val data/cifar/val` 
   - To create train and test folder : run in python console the `create_files_train_test` function from `going_modular/common.py`.
   - Open `going/modular/data_setup.py` :
     - Add normalize values specified to the dataset in the `NORMALIZE_DICT`
     - If the dataset comes from the torchvision library, add a condition similar to that of CIFAR in `load_datasets` function
   - Open `going/modular/common.py` and add condition similar to the other datasets in `classes_string`
   - When you want to run an algorithme, add the validation path if you have a specific folder for this (None by default): --data_path_val ./data/histo/val
## Classic classifier (centralized training)
### Launch the training
```
python classic.py run --data_path data/ --dataset cifar --yaml_path ./results/classic/results.yml --seed 42 --num_workers -1 --max_epochs 5 --batch_size 32 --length 32 --split 10 --device mps --save_results results/classic/ --matrix_path confusion_matrix.png --roc_path roc.png --model_save cifar.pt
```
## Federated Learning
### A. Train on local machine (Launch the simulation)
```
python simulation.py simulation --data_path data/ --dataset cifar --yaml_path ./results/FL/results.yml --seed 42 --num_workers -1 --max_epochs 5 --batch_size 32 --length 32 --split 10 --device mps --number_clients 10 --save_results results/FL/ --matrix_path confusion_matrix.png --roc_path roc.png --model_save cifar_fl.pt --min_fit_clients 10 --min_avail_clients 10 --min_eval_clients 10 --rounds 2 --frac_fit 1.0 --frac_eval 0.5
```

### B. Train without simulation
1) Run the central server

   Open a terminal windows for the central server and run the client script client.py
   ```
   python main_server.py server --data_path data/ --dataset cifar --seed 42 --num_workers 0 --max_epochs 5 --batch_size 32 --length 32 --split 10 --device mps --number_clients 3 --min_fit_clients 2 --min_avail_clients 2 --min_eval_clients 2 --rounds 2 --frac_fit 1.0 --frac_eval 0.5
   ```
2) Run each client

   Open a terminal windows for each client and run the client script client.py (change value for the client ID) 
   ```
   python main_client.py client --data_path data/ --dataset cifar --seed 42 --num_workers 0 --max_epochs 5 --batch_size 32 --length 32 --split 10 --device mps --number_clients 3 --save_results results/FL/ --matrix_path confusion_matrix2.png --roc_path roc2.png --id_client 0
   ```

## To use the homomorphic encryption
If you have the TenSEAL private/public key combination, you can use homomorphic encryption by adding the command `--he` at client and server sites. 

The private key must be on the client side and the public key on the server side, but the private key must be the same for all clients because all weights must be encrypted with the same private key. 

Otherwise, you can create the combined private/public keys on a common entity (not the aggregation server) by running the create_keys.py script: `create_keys.py` script : 
```
python create_keys.py
```

Warning :
- you have to define the path for the crypted results : `--path_crypted server.pkl` (The crypted (and not crypted) weights are saved by default in "server.pkl" file)
- server side : 
  -  you must have the public key (by default in "server_key.pkl" file) : `--path_public_key server_key.pkl`
- client side : 
  - You must have the combo private/public keys (by default in "secret.pkl" file): `--path_keys secret.pkl`

## References

The federated learning framework used is https://github.com/adap/flower and the HE library used is https://github.com/OpenMined/TenSEAL, please refer to their documentation for more information.