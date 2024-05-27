from numba import cuda
import numpy as np
import time

# Constants
N = 1_000_000_000  # Length of the array

# GPU implementation
@cuda.jit
def linear_search_gpu(arr, target, result):
    idx = cuda.grid(1)
    if idx < arr.size and arr[idx] == target:
        result[0] = idx

def main():
    # Check if sufficient GPU memory is available
    total_bytes_needed = N * np.int32(0).nbytes  # Calculate total bytes needed for the array
    free_mem = cuda.current_context().get_memory_info()[0]  # Get free memory on GPU
    
    if total_bytes_needed > free_mem:
        print("Not enough GPU memory available.")
        return

    # Generate random data
    print("Generating random data...")
    data = np.random.randint(0, 100, N, dtype=np.int32)
    
    target = np.random.randint(0, 100)  # Random target
    print(f"Target to find: {target}")

    # Transfer data to GPU
    print("Transferring data to GPU...")
    data_gpu = cuda.to_device(data)
    result_gpu = cuda.to_device(np.array([-1], dtype=np.int32))

    # GPU Search
    threads_per_block = 1024
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

    print("Starting GPU search...")
    start_time = time.time()
    linear_search_gpu[blocks_per_grid, threads_per_block](data_gpu, target, result_gpu)
    cuda.synchronize()  # Ensure all GPU work is complete
    gpu_time = time.time() - start_time
    
    index_gpu = result_gpu.copy_to_host()[0]
    print(f"GPU Search - Target found at index: {index_gpu}, Time taken: {gpu_time:.5f} seconds")

if __name__ == "__main__":
    main()
