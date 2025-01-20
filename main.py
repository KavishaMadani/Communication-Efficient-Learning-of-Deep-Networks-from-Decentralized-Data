import os
import time
import torch
import threading
from arguments import getArgs
from server import run_server
from client import run_client

args = getArgs()
os.makedirs(args.drive_path + args.exp_name, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start the server thread with the clustering_method argument
server_thread = threading.Thread(target=run_server, args=(args.clustering_method,))
server_thread.start()

# Wait for a few seconds to ensure the server is ready
time.sleep(200)  # Adjust the time as necessary

client_threads = []
for i in range(args.clients):
    client_thread = threading.Thread(target=run_client, args=(i,))
    client_thread.start()
    client_threads.append(client_thread)

server_thread.join()
for client_thread in client_threads:
    client_thread.join()

print("All clients have finished. Server has completed its execution.")