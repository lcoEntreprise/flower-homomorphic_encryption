import torchvision
import flwr as fl
import warnings
from going_modular import *
"""
The script with the common functions for the federated learning pipeline (client and server)
"""

warnings.simplefilter("ignore")

print("flwr", fl.__version__)
print("numpy", np.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################
# 1) training parameters`
main_parser2 = parsing(description='Federated Learning asset')
args = main_parser2.parse_args()
CLASSES = classes_string(args.dataset)

# In addition, we define the device allocation in PyTorch with:
DEVICE = torch.device(choice_device(args.device))
print(f"Training on {DEVICE}")

# 2) Load model and data
trainloaders, valloaders, testloader = data_setup.load_datasets(num_clients=args.number_clients,
                                                                batch_size=args.batch_size, resize=args.length,
                                                                seed=args.seed, num_workers=args.num_workers,
                                                                splitter=args.split, dataset=args.dataset,
                                                                data_path=args.data_path,
                                                                data_path_val=args.data_path_val)
