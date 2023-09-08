from going_modular import *

import warnings
warnings.simplefilter("ignore")

"""
Script to train a model on a dataset using PyTorch with the classic ML approach (no federated learning).
"""
main_parser = parsing(description='PyTorch baseline Training')
args = main_parser.parse_args()
# To define the device allocation in PyTorch:
DEVICE = torch.device(choice_device(args.device))  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__}")

if __name__ == '__main__':
    # 1) training parameters`
    number_clients = args.number_clients
    epochs = args.max_epochs
    matrix_path = args.matrix_path
    CLASSES = classes_string(args.dataset)
    NUM_CLASSES = len(CLASSES)
    yaml_path = args.yaml_path

    # 2) Load model and data
    trainloaders, valloaders, testloader = data_setup.load_datasets(num_clients=number_clients,
                                                                    batch_size=args.batch_size,
                                                                    resize=args.length, seed=args.seed,
                                                                    num_workers=args.num_workers, splitter=args.split,
                                                                    dataset=args.dataset, data_path=args.data_path,
                                                                    data_path_val=args.data_path_val)

    # To get the first trainloader and valloader (for the first client)
    trainloader = trainloaders[0]
    valloader = valloaders[0]

    # to define the model and the optimizer
    net = Net(num_classes=len(CLASSES)).to(DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # to get the trained model and the trained parameters (optimizer, metrics, ...)
    if os.path.exists(args.model_save):
        print(" To get the checkpoint")
        checkpoint = torch.load(args.model_save, map_location=DEVICE)

        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']

    else:
        # To give bad values
        acc = 0
        loss = 1000
        epoch = 0

    # to define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # 3) Train the model
    start_train = time.time()
    results = engine.train(net, trainloader, valloader, optimizer=optimizer, loss_fn=criterion, epochs=epochs,
                           device=DEVICE)

    end_train = time.time()
    # test net with another dataset
    test_loss, test_acc, y_pred, y_true, y_proba = engine.test(net, testloader, loss_fn=criterion, device=DEVICE)
    end_test = time.time()
    print(f"Final test set performance:\n\tloss {test_loss}\n\taccuracy {test_acc} %")
    print(f"Time performance : Train = {end_train - start_train}seconds  \n\tTest = {end_test - end_train} seconds")
    write_yaml({"Train_time": end_train - start_train, "Test_time": end_test - end_train},
               yaml_path,
               data1=read_yaml(yaml_path) if os.path.exists(yaml_path) else None
               )

    # Save results
    if args.save_results:
        save_graphs(args.save_results, epochs, results)

        # Build confusion matrix
        if args.matrix_path:
            save_matrix(y_true, y_pred, args.save_results + args.matrix_path, CLASSES)

        # Build roc curve
        if args.roc_path:
            save_roc(y_true, y_proba, args.save_results + args.roc_path, NUM_CLASSES)

    if args.model_save and test_acc > acc:
        print(f"save checkpoint:\nepoch: {epochs + epoch}, \t'loss: {test_loss},\tacc: {test_acc}")

        torch.save({
            'epoch': epochs + epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'acc': test_acc,
        }, args.model_save)
