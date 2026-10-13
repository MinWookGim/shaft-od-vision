# 멘토가 준 Multiresolution KD 이상 탐지(Salehi 외 2021) 코드를 지금 torch 로 옮긴 것
# 원본에서 그대로 가져온 것: 선생 VGG16(ImageNet, 고정), 학생 cfg 'B'(채널 줄인 VGG+BN),
#   학습 손실 = 층 4곳(출력 3,6,9,12번) cosine 차이 + lamda x MSE, 점수 = 층 3곳(6,9,12) 같은 식
#   위치 찾기 = 점수를 입력 픽셀로 미분한 값(gradients 방식) + 가우시안 흐림 sigma 4
# 바꾼 것: vgg16 가중치 부르는 방식(새 torchvision), .cuda() 강제 -> device 인자
import torch, numpy as np
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from scipy.ndimage import gaussian_filter

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList(list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features)).eval()
        for p in self.parameters(): p.requires_grad_(False)
    def forward(self, x):
        out = []
        for i in range(31):
            x = self.features[i](x)
            if i in (1, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30): out.append(x)
        return out

class Student(nn.Module):
    CFG = [16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M', 16, 16, 512, 'M']   # 원본 cfg 'B'
    def __init__(self):
        super().__init__()
        layers, c = [], 3
        for i, v in enumerate(self.CFG):
            if v == 'M': layers += [nn.MaxPool2d(2, 2)]; continue
            conv = nn.Conv2d(c, v, 3, padding=1, bias=False); nn.init.xavier_uniform_(conv.weight)
            layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] if self.CFG[i + 1] != 'M' else [conv, nn.ReLU(inplace=True)]
            c = v
        self.features = nn.Sequential(*layers)
    def forward(self, x):
        out = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38): out.append(x)
        return out

cos = nn.CosineSimilarity()
def loss_terms(sp, tp, idx, lamda):
    """사진마다 따로 계산한 점수(배치 평균 아님)"""
    tot = 0
    for k in idx:
        a, b = sp[k].flatten(1), tp[k].flatten(1)
        tot = tot + (1 - cos(a, b)) + lamda * ((a - b) ** 2).mean(1)
    return tot

def train(student, teacher, loader_fn, epochs, lamda, dev, lr=1e-3, log=None):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for ep in range(epochs):
        student.train(); tot = 0
        for x in loader_fn():
            x = x.to(dev)
            l = loss_terms(student(x), teacher(x), (3, 6, 9, 12), lamda).mean()
            opt.zero_grad(); l.backward(); opt.step(); tot += l.item()
        if log and (ep % 50 == 0 or ep == epochs - 1): log(f'epoch {ep} loss {tot:.4f}')

@torch.no_grad()
def score(student, teacher, x, lamda, dev):
    student.eval()
    return loss_terms(student(x.to(dev)), teacher(x.to(dev)), (6, 9, 12), lamda).cpu().numpy()

def heatmap(student, teacher, x, lamda, dev):
    student.eval(); x = x.to(dev).clone().requires_grad_(True)
    s = loss_terms(student(x), teacher(x), (6, 9, 12), lamda).sum()
    g, = torch.autograd.grad(s, x)
    g = g.abs().sum(1).cpu().numpy()
    return np.stack([gaussian_filter(m, 4) for m in g])
