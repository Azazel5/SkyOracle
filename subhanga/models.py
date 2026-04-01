import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self.skip: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.skip(x)
        return self.act(out)


class WeatherResNetGAP(nn.Module):
    """
    CNN that outputs 6 continuous targets (evaluator-compatible).

    Input:  (B, 42, 450, 449) float32
    Output: (B, 6) float32 (in whatever target space the training script uses)
    """

    def __init__(self, in_channels: int = 42, base_channels: int = 32, out_dim: int = 6) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = ConvBNAct(in_channels, c1, kernel_size=3, stride=1)
        self.stage1 = nn.Sequential(BasicBlock(c1, c1), BasicBlock(c1, c1))
        self.stage2 = nn.Sequential(BasicBlock(c1, c2, stride=2), BasicBlock(c2, c2))
        self.stage3 = nn.Sequential(BasicBlock(c2, c3, stride=2), BasicBlock(c3, c3))
        self.stage4 = nn.Sequential(BasicBlock(c3, c4, stride=2), BasicBlock(c4, c4))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class DenormalizeTargets(nn.Module):
    """
    Wraps a base model that predicts normalized targets and converts to real units.
    """

    def __init__(self, base: nn.Module, y_mean: torch.Tensor, y_std: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("y_mean", y_mean.float().clone())
        self.register_buffer("y_std", y_std.float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_norm = self.base(x)
        return y_norm * self.y_std + self.y_mean
