"""
分类网络MOE
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeModel(nn.Module):
    def __init__(self, input_size, vocab_size=63, num_mixtures=4):
        super(MoeModel, self).__init__()
        self.gate_fc = nn.Linear(input_size, vocab_size*(num_mixtures+1))
        self.expert_fc = nn.Linear(input_size, vocab_size*num_mixtures)
        self.vocab_size = vocab_size
        self.num_mixtures = num_mixtures
        self.model_name = "MoeModel"

    def forward(self, x):
        expert_x = self.expert_fc(x)
        expert_x = expert_x.view(-1, self.num_mixtures)  # 此时expert_x的维度是[batch*vocab_size]
        expert_x = F.sigmoid(expert_x)
        gate_x = self.gate_fc(x)
        gate_x = gate_x.view(-1, self.num_mixtures + 1)
        gate_x = F.softmax(gate_x, dim=1)
        final_proba = torch.sum(gate_x[:, :self.num_mixtures] * expert_x, 1).view(-1, self.vocab_size)
        return final_proba

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def save(self, name=None):
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name


if __name__ == "__main__":
    '''
    model = MoeModel(512)
    for name, param in model.named_parameters():
        print(name, param.size())
    print(list(model.children()))
    '''
    '''
    model = MoeModel(512)
    data = torch.ones(2, 512)
    output = model(data)
    print(output)
    print(output.shape)
    '''
    print(callable(MoeModel))
