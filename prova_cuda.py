import torch
import torch.nn as nn

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:

    # Check CUDA device
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Device 0: {torch.cuda.get_device_name(0)}")

    # Create a tensor and move it to the GPU
    tensor = torch.tensor([1.0, 2.0, 3.0])
    tensor = tensor.to('cuda')

    # Perform a simple operation on the GPU
    result = tensor * 2

    # Print the result
    print(result)

    print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
    print(f"Memory Cached: {torch.cuda.memory_reserved()} bytes")
    print(f"====================================================")
    print(f"====================================================")
    print(f"====================================================")

    # Define a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(3, 3)
        
        def forward(self, x):
            x = self.fc1(x)
            return x

    # Create an instance of the neural network
    model = SimpleNN().to('cuda')

    # Create a random input tensor and move it to the GPU
    input_tensor = torch.randn(1, 3).to('cuda')

    # Perform a forward pass through the network
    output = model(input_tensor)

    # Print the output
    print(output)

    # Check memory usage again
    print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
    print(f"Memory Cached: {torch.cuda.memory_reserved()} bytes")
