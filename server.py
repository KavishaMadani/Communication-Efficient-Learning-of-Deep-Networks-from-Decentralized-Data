import os
from communication import Communication
import threading
import csv
import torch
from model import MOON
import main_datasets
from update_global_model import update_global_model
from clusters import create_clusters
from unbiased import create_unbiased_model
from accuracy import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score
from data_distribution import data_distribution
from arguments import getArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = getArgs()

# Construct file paths using os.path.join for better compatibility
drive_path = args.drive_path
exp_name = args.exp_name

# Create necessary directories
os.makedirs(os.path.join(drive_path, exp_name, 'client_datasets'), exist_ok=True)
os.makedirs(os.path.join(drive_path, exp_name, 'models'), exist_ok=True)
os.makedirs(os.path.join(drive_path, exp_name, 'logs'), exist_ok=True)

# Define model paths
global_model_path = os.path.join(drive_path, exp_name, 'models', 'global_model.pth')
global_optimizer_path = os.path.join(drive_path, exp_name, 'models', 'global_optimizer.pth')
model_clusters_path = os.path.join(drive_path, exp_name, 'models', 'model_clusters.pth')
unbiased_model_path = os.path.join(drive_path, exp_name, 'models', 'unbiased_model.pth')

def process_client(conn, comm):
    print("sending signal to client...")
    comm.send_signal("Start training", conn)
    print("signal sent to client")
    
    print("waiting....")
    signal=comm.recieve_signal(conn)
    print(f'signal: {signal} recieved')

# Function to run the server
def run_server(clustering_method):
    comm = Communication(host='127.0.0.1', port=9999)

    with open(os.path.join(drive_path, exp_name, 'logs', f"{args.dataset}_log_server.csv"), 'w', newline='') as csvfile:
        fieldnames = ['round', 'training_accuracy', 'testing_accuracy', 'precision', 'recall', 'f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    trainset = ""
    testset = ""
 
    if args.dataset == 'mnist':
        trainset = main_datasets.load_mnist_dataset(train=True)
        testset = main_datasets.load_mnist_dataset(train=False)
    elif args.dataset == 'fmnist':
        trainset = main_datasets.load_fmnist_dataset(train=True)
        testset = main_datasets.load_fmnist_dataset(train=False)
    elif args.dataset == 'cifar10':
        trainset = main_datasets.load_cifar10_dataset(train=True)
        testset = main_datasets.load_cifar10_dataset(train=False)
    elif args.dataset == 'svhn':
        trainset = main_datasets.load_svhn_dataset(train=True)
        testset = main_datasets.load_svhn_dataset(train=False)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)
  
    global_model = MOON().to(device)
    model_clusters={}
    unbiased_model = MOON().to(device)

    datasets = data_distribution(trainset, -1, 10, args.distribution, args.clients, 0.5, 42, drive_path + exp_name)
    data_sizes = [len(dataset) for dataset in datasets.values()]

    comm.init_server()
    print("--------------------------------------------------------------server running--------------------------------------------------------------")

    connections = []
    for i in range(args.clients):
        conn = comm.server_accept()
        connections.append(conn)

    for round in range(args.rounds):
        local_models = []
        threads = []
        global_model.train()

        torch.save(global_model.state_dict(), global_model_path)
        torch.save(unbiased_model.state_dict(), unbiased_model_path)
        torch.save(model_clusters, model_clusters_path)    

        for conn in connections:
            thread = threading.Thread(target=process_client, args=(conn, comm))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        for i in range(args.clients):
            local_model = MOON().to(device)
            local_model.load_state_dict(torch.load(os.path.join(drive_path, exp_name, f'client_datasets/client_{i}_model.pt')))
            local_models.append(local_model)

        global_model = update_global_model(global_model, local_models, data_sizes)

        # Use the specified clustering method
        model_clusters = create_clusters(local_models, args.clusters, model_name=clustering_method)
        unbiased_model = create_unbiased_model(model_clusters)
        

        train_accuracy = calculate_accuracy(trainloader, global_model)
        test_accuracy = calculate_accuracy(testloader, global_model)

        # Calculate precision, recall, and F1 score
        precision = calculate_precision(global_model, testloader)
        recall = calculate_recall(global_model, testloader)
        f1_score = calculate_f1_score(precision, recall)

        # Print results for the specified clustering method
        print(f"Results for {clustering_method} clustering method:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        print("-" * 60)  # Separator for clarity

        # Log results for the specified clustering method
        with open(os.path.join(drive_path, exp_name, 'logs', f"{args.dataset}_log_{clustering_method}_server.csv"), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'round': round + 1,
                'training_accuracy': train_accuracy,
                'testing_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })

    # Final accuracy calculations
    train_accuracy = calculate_accuracy(trainloader, global_model)
    test_accuracy = calculate_accuracy(testloader, global_model)
    print(f"Final Train Accuracy on server with {clustering_method}: ", train_accuracy)
    print(f"Final Test Accuracy on server with {clustering_method}: ", test_accuracy)

    for conn in connections:
        comm.close_connection(conn)
    comm.close_server()

# Call the run_server function to start the server
if __name__ == "__main__":
    clustering_method = args.clustering_method  # Get the clustering method from arguments
    run_server(clustering_method)