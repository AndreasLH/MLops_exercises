import time
from torchvision import models
import torch
from ptflops import get_model_complexity_info

m1 = models.efficientnet_b3(weigths=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
m2 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
m3 = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

input = torch.randn(16, 3, 256, 256)
n_reps = 3
# m1.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# model_fp32_prepared = torch.quantization.prepare(m1)
# model_int8 = torch.quantization.convert(model_fp32_prepared)
model_int8 = torch.quantization.quantize_dynamic(
    m1,  # the original model
    dtype=torch.qint8)

macs, params = get_model_complexity_info(m1, (3, 256, 256), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

macs, params = get_model_complexity_info(m2, (3, 256, 256), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

macs, params = get_model_complexity_info(m3, (3, 256, 256), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

for i, m in enumerate([m1, model_int8, m2, m3]):
   tic = time.perf_counter()
   for _ in range(n_reps):
      _ = m(input)
   toc = time.perf_counter()
   print(f"Model {i} took: {(toc - tic) / n_reps}")