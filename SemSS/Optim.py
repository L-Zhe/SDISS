import  torch
from    torch import nn
from    torch.nn import functional as F

class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0., lamda=0.):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.smoothing = smoothing
        self.lamda     = lamda

    def forward(self, inputs, targets, penalty, PAD):
        # inputs = F.log_softmax(inputs, dim=-1)
        vocab_size = inputs.size(-1)
        batch_size = targets.size(0)
        length     = targets.size(1)
        mask       = (targets == PAD).view(-1)
        index      = targets.unsqueeze(-1)
        targets    = torch.zeros(batch_size, length, vocab_size)\
                       .to(targets.device).scatter(-1, index, 1)
        targets = targets * (1 - self.smoothing) + self.smoothing / vocab_size
        loss    = self.criterion(inputs.view(-1, vocab_size), 
                              targets.view(-1, vocab_size).detach()).sum(dim=-1) + self.lamda * penalty.view(-1)
        return loss.masked_fill(mask, 0.).sum()/batch_size



class WarmUpOpt:

    def __init__(self, optimizer, d_model, warmup_steps, factor=1):
        self.__step       = 0
        self.factor       = factor
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.optimizer    = optimizer

    def updateRate(self):
        return self.factor * (self.d_model**(-0.5) * 
                    min(self.__step**(-0.5), self.__step * self.warmup_steps**(-1.5)))

    def step(self):
        self.__step += 1
        for param in self.optimizer.param_groups:
            param['lr'] = self.updateRate()
        self.optimizer.step()
        self.optimizer.zero_grad()
        