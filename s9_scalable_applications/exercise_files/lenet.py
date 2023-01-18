import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import prune
import time

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = LeNet()

    # print all the parameters of the model
    for m in model.named_modules():
        print(m)

    # print parameters of the first module
    module_1 = model.conv1
    print(list(module_1.named_parameters()))
    print(list(module_1.named_buffers()))

    # module_1 = model.conv1
    # prune.random_unstructured(module_1, name="weight", amount=0.3)
    # print("pruned model")
    # print(list(module_1.named_parameters()))
    # print(list(module_1.named_buffers()))
    module_1 = model.conv1
    prune.l1_unstructured(module_1, name="bias", amount=0.3)
    print("pruned model")
    print(list(module_1.named_parameters()))
    print(list(module_1.named_buffers()))

    new_model = LeNet()
    name, module = next(new_model.named_modules())
    prune.l1_unstructured(module.conv1, name='weight', amount=0.2)
    prune.l1_unstructured(module.conv2, name='weight', amount=0.2)
    prune.random_unstructured(module.fc1, name="weight", amount=0.4)
    prune.random_unstructured(module.fc2, name="weight", amount=0.4)
    prune.random_unstructured(module.fc3, name="weight", amount=0.4)

    print(dict(new_model.named_buffers()).keys())

    model = LeNet()

    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    def check_prune_level(module: nn.Module):
        sparsity_level = 100 * float(torch.sum(module.weight == 0) / module.weight.numel())
        print(f"Sparsity level of module {sparsity_level}")

    check_prune_level(model.conv1)
    check_prune_level(model.conv2)
    check_prune_level(model.fc1)
    check_prune_level(model.fc2)
    check_prune_level(model.fc3)
    
    for module, type in parameters_to_prune:
        module.weight = torch.nn.Parameter(module.weight.data.to_sparse())
        prune.remove(module, name='weight')

    network = LeNet()
    tic = time.time()
    for _ in range(100):
        _ = network(torch.randn(100, 1, 28, 28))
    toc = time.time()
    print('unpruned model', toc - tic)
    
    tic = time.time()
    for _ in range(100):
        _ = model(torch.randn(100, 1, 28, 28))
    toc = time.time()
    print('pruned model',toc - tic)

    torch.save(model.state_dict(), 'pruned_network.pt')
    torch.save(network.state_dict(), 'network.pt')