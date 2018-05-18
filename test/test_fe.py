
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.feature_extractors.faster_rcnn_feature_extractors_api import FasterRCNNFeatureExtractors


class TestFasterRCNNFeatureExtractors(unittest.TestCase):
    class FlatNet(nn.Module):
        def __init__(self):
            super(TestFasterRCNNFeatureExtractors.FlatNet, self).__init__()
            self.pool = nn.MaxPool2d(2, 2)
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class FlatNetSequential(nn.Module):
        def __init__(self):
            super(TestFasterRCNNFeatureExtractors.FlatNetSequential, self).__init__()
            self.modul = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.Linear(16 * 5 * 5, 120),
                nn.Linear(120, 84),
                nn.Linear(84, 10),
            )

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def create_flat_net(self):
        N = 100
        x = Variable(torch.randn(N, 3, 32, 32))
        y = Variable(torch.randn(N, 10))
        model = self.FlatNet()
        return model, x, y

    def run_model(self, model, x, y, freeze_upto_pooling_num):
        loss_fn = torch.nn.MSELoss(size_average=False)

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        FasterRCNNFeatureExtractors._freeze_layers(model, freeze_upto_pooling_num)

        for t in range(10):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)

            loss = loss_fn(y_pred, y)

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        return model

    def clone(self, model):
        named_parameters = {}
        for name, param in model.named_parameters():
            named_parameters[name] = param.clone()
        return named_parameters

    def run_test(self, freeze_upto, layer_freeze_stop):
        model, x, y = self.create_flat_net()
        initial_parameters = self.clone(model)

        model = self.run_model(model, x, y, freeze_upto)

        for name, param in model.named_parameters():
            param_name = name.split("_")
            if int(param_name[1].split(".")[0]) <= layer_freeze_stop:
                self.assertTrue(torch.equal(param, initial_parameters[name]), "assertTrue of {}".format(name))
            else:
                self.assertFalse(torch.equal(param, initial_parameters[name]), "assertFalse of {}".format(name))


    def test_freeze_layers(self):
        self.run_test(0, 0)
        self.run_test(1, 1)
        self.run_test(2, 6)

if __name__ == '__main__':
    unittest.main()
