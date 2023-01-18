import torch

input = torch.randn(3)

out = torch.quantize_per_tensor(input, 0.1, 0, torch.quint8)
out2 = out.dequantize()
a = 2 