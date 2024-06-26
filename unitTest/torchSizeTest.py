import torch
import sys
epsilon = 1e-5
tensor = torch.tensor(epsilon)

print(tensor.size())

size_of_epsilon = sys.getsizeof(epsilon)

print(size_of_epsilon)
