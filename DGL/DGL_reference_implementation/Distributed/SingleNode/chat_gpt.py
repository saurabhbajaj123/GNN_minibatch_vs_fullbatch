import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    # Clean up the process group
    dist.destroy_process_group()

def train(rank, model, criterion, optimizer, train_loader, num_epochs, convergence_threshold):
    # Set up the distributed training
    setup(rank, num_gpus)

    # Use DistributedDataParallel to wrap the model
    model = DDP(model.to(rank), device_ids=[rank])

    # Synchronize the model parameters across processes
    torch.cuda.synchronize()

    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Synchronize the loss across processes
        avg_loss_tensor = torch.tensor(avg_loss).to(rank)
        dist.all_reduce(avg_loss_tensor)
        avg_loss = avg_loss_tensor.item() / num_gpus

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Check convergence based on error range around the final value
        if epoch > 0:
            if abs(avg_loss - prev_loss) < convergence_threshold:
                break
        prev_loss = avg_loss

    end_time = time.time()
    total_time = end_time - start_time

    # Synchronize the total time across processes
    total_time_tensor = torch.tensor(total_time).to(rank)
    dist.all_reduce(total_time_tensor)
    total_time = total_time_tensor.item()

    # Synchronize the number of epochs across processes
    num_epochs_tensor = torch.tensor(epoch + 1).to(rank)
    dist.all_reduce(num_epochs_tensor)
    num_epochs = num_epochs_tensor.item()

    print(f"Rank {rank}: Time taken for convergence: {total_time:.2f} seconds")
    print(f"Rank {rank}: Number of epochs: {num_epochs}")

    cleanup()

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    world_size = num_gpus
    convergence_threshold = 0.001  # Adjust this threshold as per your requirement

    mp.spawn(train, args=(model, criterion, optimizer, train_loader, num_epochs, convergence_threshold),
             nprocs=num_gpus, join=True)
