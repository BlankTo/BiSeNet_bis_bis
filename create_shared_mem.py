import os

shared_mem_path = os.path.join("shared_mem", "file.bin")

size_bytes = 8 * 1024 * 1024 * 1024  # Convert GB to bytes

# Create the file and write data to it
with open(shared_mem_path, 'wb') as f:
    # Write zeros to fill the file and create the specified size
    for _ in range(size_bytes // (1024 * 1024)):
        f.write(b'\x00' * (1024 * 1024))  # Write 1 MB of zeros at a time

print(f"Created file: {shared_mem_path}")