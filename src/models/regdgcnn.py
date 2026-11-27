"""B3: RegDGCNN for per-point surface pressure, the baseline of the DrivAerNet++ authors.
Adapted from DrivAerNet (Elrefaie et al.), RegDGCNN_SurfaceFields/model_pressure.py, CC BY-NC 4.0: layers and forward pass unchanged
(EdgeConv x3 with k nearest neighbours in feature space, global max embedding, per-point MLP head). Only the interface differs:
it takes our batch (normalised query coordinates `ndc`) and returns (B, N). Their defaults: k = 40, emb_dims = 1024, dropout 0.4, 10,000 points."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]  # (B, N, k)


def get_graph_feature(x, k=20):
    b, _, n = x.shape
    idx = knn(x, k) + torch.arange(0, b, device=x.device).view(-1, 1, 1) * n
    d = x.size(1)
    xt = x.transpose(2, 1).contiguous()
    feat = xt.view(b * n, -1)[idx.view(-1), :].view(b, n, k, d)
    xt = xt.view(b, n, 1, d).repeat(1, 1, k, 1)
    return torch.cat((feat - xt, xt), dim=3).permute(0, 3, 1, 2).contiguous()  # (B, 2d, N, k)


def cbl2(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))


def cbl1(i, o):
    return nn.Sequential(nn.Conv1d(i, o, 1, bias=False), nn.BatchNorm1d(o), nn.LeakyReLU(0.2))


class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = cbl2(6, 64), cbl2(64, 128), cbl1(128, 1024)
        self.linear1, self.bn3 = nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512)
        self.linear2, self.bn4 = nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256)
        self.transform = nn.Linear(256, 9)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        x = self.conv2(self.conv1(x)).max(dim=-1)[0]
        x = self.conv3(x).max(dim=-1)[0]
        x = F.leaky_relu(self.bn3(self.linear1(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.linear2(x)), 0.2)
        return self.transform(x).view(x.size(0), 3, 3)


class RegDGCNN(nn.Module):
    def __init__(self, k=40, emb_dims=1024, dropout=0.4):
        super().__init__()
        self.k = k
        self.transform_net = TransformNet()
        self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 = cbl2(6, 64), cbl2(64, 64), cbl2(128, 64), cbl2(64, 64), cbl2(128, 64)
        self.conv6 = cbl1(192, emb_dims)
        self.conv8, self.conv9, self.conv10 = cbl1(emb_dims + 192, 256), cbl1(256, 256), cbl1(256, 128)
        self.dp1, self.dp2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.conv11 = nn.Conv1d(128, 1, 1, bias=False)

    def forward(self, vol, g, ndc, nrm):
        x = ndc.transpose(1, 2).float()  # (B, 3, N) normalised coordinates
        n = x.size(2)
        t = self.transform_net(get_graph_feature(x, self.k))
        x = torch.bmm(x.transpose(2, 1), t).transpose(2, 1)
        x1 = self.conv2(self.conv1(get_graph_feature(x, self.k))).max(dim=-1)[0]
        x2 = self.conv4(self.conv3(get_graph_feature(x1, self.k))).max(dim=-1)[0]
        x3 = self.conv5(get_graph_feature(x2, self.k)).max(dim=-1)[0]
        e = self.conv6(torch.cat((x1, x2, x3), 1)).max(dim=-1, keepdim=True)[0].repeat(1, 1, n)
        h = self.dp2(self.conv9(self.dp1(self.conv8(torch.cat((e, x1, x2, x3), 1)))))
        return self.conv11(self.conv10(h)).squeeze(1)  # (B, N)
