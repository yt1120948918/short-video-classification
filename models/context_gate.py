from torch import nn


class ContextGate(nn.Module):
    def __init__(self, input_size):
        super(ContextGate, self).__init__()
        self.gate = nn.Linear(input_size, input_size)

    def forward(self, x):
        gate = self.gate(x)
        x = x * gate
        return x

