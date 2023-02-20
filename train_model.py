from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 1
    lrate = 0.001
    loss_function = nn.MSELoss()
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)


    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)


    for epoch_i in range(no_epochs):
        model.train()
        tl = 0
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            optimizer.zero_grad()
            loss = loss_function(model(sample['input']), sample['label'].unsqueeze(1))
            loss.backward()
            optimizer.step()
            tl += loss.item()
        losses.append(tl)
    torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)



if __name__ == '__main__':
    no_epochs = 25
    train_model(no_epochs)
