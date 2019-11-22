import torch.nn as nn
import torch

class ProtoNetBig(nn.Module):
    def __init__(self, x_dim=23433, hid_dim=[2000, 1000, 500, 250], z_dim=100):
        super(ProtoNetBig, self).__init__()
        self.linear0 = nn.Linear(x_dim, hid_dim[0])
        self.bn1 = nn.BatchNorm1d(hid_dim[0])
        self.linear1 = nn.Linear(hid_dim[0], hid_dim[1])
        self.bn2 = nn.BatchNorm1d(hid_dim[1])
        self.linear2 = nn.Linear(hid_dim[1] + hid_dim[0], hid_dim[2])
        self.bn3 = nn.BatchNorm1d(hid_dim[2])
        self.linear3 = nn.Linear(hid_dim[1] + hid_dim[0] + hid_dim[2], hid_dim[3])
        self.bn4 = nn.BatchNorm1d(hid_dim[3])
        self.linear4 = nn.Linear(hid_dim[1] + hid_dim[0] + hid_dim[2] + hid_dim[3], z_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(inplace=True)
    def forward(self, x):
        out = self.dropout(self.bn1(self.relu(self.linear0(x))))
        out1 = self.dropout(self.bn2(self.relu(self.linear1(out))))
        out2 = torch.cat([out, out1], 1)
        out3 = self.dropout(self.bn3(self.relu(self.linear2(out2))))
        out4 = torch.cat([out, out1, out3], 1)
        out5 = self.dropout(self.bn4(self.relu(self.linear3(out4))))
        out6 = torch.cat([out, out1, out3, out5], 1)
        out7 = self.linear4(out6)
        return out7