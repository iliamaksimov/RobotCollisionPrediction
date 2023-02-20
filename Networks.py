import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self, inpsize=6, hidsize=200, outsize=1):

        super().__init__()
        self.inp_hid = nn.Linear(inpsize, hidsize)
        self.nonl_act = nn.ReLU()
        self.hid_out = nn.Linear(hidsize, outsize)


    def forward(self, input):

        hid = self.inp_hid(input)
        hid = self.nonl_act(hid)
        output = self.hid_out(hid)
        return output


    def evaluate(self, model, test_loader, loss_function):

        loss = 0
        for sample in test_loader:
            n_out = model(sample['input'])
            l = loss_function(n_out, sample['label'].unsqueeze(1))
            loss += l.item()
        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
