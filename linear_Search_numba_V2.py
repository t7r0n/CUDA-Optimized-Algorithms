from numba import cuda
import numpy as np
import time

# Constants
N = 1_000_000_000_000  # Trillion elements

# Define GPU Linear Search Kernel
@cuda.jit
def linear_search_gpu(arr, target, result):
    idx = cuda.grid(1)
    if idx < arr.size and arr[idx] == target:
        result[0] = idx

# CPU Fallback Linear Search for validation or small chunks
def linear_search_cpu(arr, target):
    for i in range(arr.size):
        if arr[i] == target:
            return i
    return -1

def search_chunk(chunk, target):
    data_gpu = cuda.to_device(chunk)
    result_gpu = cuda.to_device(np.array([-1], dtype=np.int32))

    threads_per_block = 1024
    blocks_per_grid = (chunk.size + (threads_per_block - 1)) // threads_per_block

    linear_search_gpu[blocks_per_grid, threads_per_block](data_gpu, target, result_gpu)
    cuda.synchronize()
    result = result_gpu.copy_to_host()[0]

    if result != -1:
        return result

    return -1

def main():
    # Calculate available GPU memory and determine chunk size
    free_mem = cuda.current_context().get_memory_info()[0]  # Get free memory on GPU
    element_size = np.int32(0).nbytes  # Size of each element
    chunk_size = free_mem // element_size  # Number of elements that fit in the available GPU memory

    if chunk_size == 0:
        print("Not enough GPU memory available to process any chunk.")
        return

    # Ensure chunk size is reasonable for processing
    chunk_size = min(chunk_size, N)  # Cap chunk size to the total number of elements
    num_chunks = (N + chunk_size - 1) // chunk_size  # Calculate the number of chunks needed

    print(f"Free GPU memory: {free_mem / (1024 ** 3):.2f} GB")
    print(f"Element size: {element_size} bytes")
    print(f"Chunk size: {chunk_size} elements")
    print(f"Number of chunks needed: {num_chunks}")

    # Target to search for
    target = np.random.randint(0, 100)
    print(f"Target to find: {target}")

    found_index = -1

    # Process each chunk
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        current_chunk_size = chunk_end - chunk_start

        # Generate random data for the current chunk
        print(f"Processing chunk from {chunk_start} to {chunk_end}...")
        chunk = np.random.randint(0, 100, current_chunk_size, dtype=np.int32)

        # Perform GPU search on the current chunk
        start_time = time.time()
        index_in_chunk = search_chunk(chunk, target)
        end_time = time.time()

        if index_in_chunk != -1:
            found_index = chunk_start + index_in_chunk
            break

        print(f"Chunk processed in {end_time - start_time:.5f} seconds.")

    if found_index == -1:
        print("Target not found.")
    else:
        print(f"Target found at index: {found_index}")

if __name__ == "__main__":
    main()
