# GPU-Accelerated Linear Search

This repository contains an implementation of a GPU-accelerated linear search algorithm designed to handle extremely large datasets efficiently. Using NVIDIA's CUDA technology through Numba, this project demonstrates how modern GPU computing can vastly improve the performance of a simple linear search algorithm, traditionally a time-consuming operation for large arrays.

## Overview

The linear search algorithm is a basic search technique that checks every element in an array until it finds the target value. While straightforward, it becomes computationally expensive as the dataset size increases. This project leverages the parallel processing capabilities of GPUs to accelerate this process, making it feasible to handle datasets as large as 1 billion elements.

## Features

- **Massive Dataset Handling**: Capable of searching through 1 billion integers.
- **High Performance**: Achieves substantial speed improvements over traditional CPU-based searches. Preliminary tests show up to 100 times faster performance compared to sequential CPU implementations.
- **CUDA Optimization**: Utilizes CUDA's parallel execution model to maximize data throughput and minimize search time.
- **Memory Efficiency**: Includes checks to ensure the GPU memory is sufficient before attempting the search, preventing crashes and memory errors.

## Requirements

- **Python 3.8+**
- **Numba** with CUDA support
- **Numpy**
- **An NVIDIA GPU with at least 4GB of memory** (preferably 8GB or more for the best performance)

## Setup

To run this project, follow these steps:

1. Clone the repository:

  ```bash
  git clone https://github.com/t7r0n/CUDA-Optimized-Algorithms.git
   ```
2. Install the required Python packages:

```
pip install numba numpy
```

3. Execute the script:

```
python linear_Search_numba_V1.py
```

## Usage
Modify the main() function in linear_Search_numba_V1.py to set the desired array size (N) and target value. By default, the script will search for a randomly selected integer within an array of 1 billion integers.

## Benchmark Results

The following benchmark results illustrate the performance gain of using GPU acceleration:

- CPU (Intel i7-9750h): ~15 minutes
- GPU (NVIDIA RTX 2070 MaxQ): ~9 seconds
This represents more than a 100-fold increase in performance, showcasing the potential of GPU computing in data-intensive applications.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Acknowledgements
This project utilizes technologies from NVIDIA CUDA and Numba. Special thanks to the CUDA development team and the contributors of the Numba library for making high-performance GPU computing accessible to developers.

