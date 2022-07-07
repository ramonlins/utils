import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init


class VGGReg(torch.nn.Module):
    """
    VGGReg is a network to estimate the density of the tree. It is a VGG model
    with conections to the end of the network.
    """

    def __init__(self, input_res, pretrained=True):
        """Build a vgg16 with skip connections

        Args:
            input_res (tuple[int, int]): Resolution of input image (w, h).
            pretrained (bool, optional): Load model with pretrained weights. Defaults to True.
        """
        super(VGGReg, self).__init__()

        self.res = input_res
        vgg16pre = torchvision.models.vgg16(pretrained=pretrained)
        self.vgg0 = torch.nn.Sequential(*list(vgg16pre.features.children())[:4])
        self.vgg1 = torch.nn.Sequential(*list(vgg16pre.features.children())[4:9])
        self.vgg2 = torch.nn.Sequential(*list(vgg16pre.features.children())[9:16])
        self.vgg3 = torch.nn.Sequential(*list(vgg16pre.features.children())[16:23])
        self.vgg4 = torch.nn.Sequential(*list(vgg16pre.features.children())[23:30])

        self.reduce_channels = torch.nn.Conv2d(512, 256, 1)

        self.smooth0 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, 1),
                torch.nn.ReLU(True)
                )
        self.smooth1 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 64, 3, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, 1),
                torch.nn.ReLU(True)
                )
        self.smooth2 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 64, 3, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, 1),
                torch.nn.ReLU(True)
                )
        self.smooth3 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 64, 3, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, 1),
                torch.nn.ReLU(True)
                )

        self.max_pool = torch.nn.MaxPool2d(2, 2)

        self.classifier = torch.nn.Sequential(
                torch.nn.Linear((input_res[0] // 32) * (input_res[1] // 32) * 512, 256),
                torch.nn.ReLU(True),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(True),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid()
                )

    def forward(self, x):
        """
        Args:
            x (torch.tensor): A tensor of size (batch, 3, H, W)
        Returns:
            reg_out (torch.tensor): A tensor with results of the regression (batch, 4).
            cls_out (torch.tensor): A tensor with results of the classification (batch, 2).
        """
        feat0 = self.vgg0(x)
        feat1 = self.vgg1(feat0)
        feat2 = self.vgg2(feat1)
        feat3 = self.vgg3(feat2)
        feat4 = self.vgg4(feat3)
        rfeat4 = self.reduce_channels(feat4)
        feat = self.max_pool(rfeat4)

        size_last = (self.res[1] // 32, self.res[0] // 32)
        res_feat0 = torch.nn.functional.interpolate(feat0, size=size_last, mode='bilinear', align_corners=True)
        res_feat0 = self.smooth0(res_feat0)

        res_feat1 = torch.nn.functional.interpolate(feat1, size=size_last, mode='bilinear', align_corners=True)
        res_feat1 = self.smooth1(res_feat1)

        res_feat2 = torch.nn.functional.interpolate(feat2, size=size_last, mode='bilinear', align_corners=True)
        res_feat2 = self.smooth2(res_feat2)

        res_feat3 = torch.nn.functional.interpolate(feat3, size=size_last, mode='bilinear', align_corners=True)
        res_feat3 = self.smooth3(res_feat3)

        feat = torch.cat([feat, res_feat0, res_feat1, res_feat2, res_feat3], 1)

        B, C, H ,W = feat.size()
        feat = feat.view(B, -1)
        output = self.classifier(feat)

        return output

    def calculate_loss(self, pred, gt, bound=0.05):
        """
        compute loss with error limiting.

        Args:
            pred (torch.tensor): A tensor (float32) of size (batch) with the ground truth [0, 1].
            gt (torch.tensor): A tensor (float32) of size (batch) with the ground truth [0, 1].
            bound (float): Defines the lower-bound and upper-bound of error limitation. Defaults to 0.05.

        Returns:
            loss (torch.tensor): A tensor with total loss.
        """
        error = pred - gt
        limit = torch.clamp(error, -bound, +bound)

        return torch.nn.functional.smooth_l1_loss(pred - limit, gt)

    @staticmethod
    def eval_net_with_loss(model, inp, gt):
        """
        Evaluate network including loss.

        Args:
            model (torch.nn.Module): The model.
            inp (torch.tensor): A tensor (float32) of size (batch, 3, H, W)
            gt (torch.tensor): A tensor (float32) of size (batch) with the groud truth [0, 1].

        Returns:
            out (torch.tensor):  A tensor (float32) of size (batch) with the groud truth [0, 1].
            loss (torch.tensor): Tensor with the total loss.

        """
        out = model(inp)
        out = torch.flatten(out)

        loss = model.calculate_loss(out, gt)
        return (out, loss)

    @staticmethod
    def get_params_by_kind(model, n_base = 7):

        base_vgg_bias = []
        base_vgg_weight = []
        core_weight = []
        core_bias = []

        for name, param in model.named_parameters():
            if 'vgg' in name and ('weight' in name or 'bias' in name):
                vgglayer = int(name.split('.')[0][-1])

                if vgglayer <= n_base:
                    if 'bias' in name:
                        base_vgg_bias.append(param)
                    else:
                        base_vgg_weight.append(param)
                else:
                    if 'bias' in name:
                        core_bias.append(param)
                    else:
                        core_weight.append(param)

            elif ('weight' in name or 'bias' in name):
                if 'bias' in name:
                    core_bias.append(param)
                else:
                    core_weight.append(param)

        return (base_vgg_weight, base_vgg_bias, core_weight, core_bias)

    def init_params(self):
        self.init_params_(self.classifier, False)
        self.init_params_(self.smooth3, False)
        self.init_params_(self.smooth2, False)
        self.init_params_(self.smooth1, False)
        self.init_params_(self.smooth0, False)
        init.kaiming_normal_(self.reduce_channels.weight, mode='fan_in', nonlinearity='relu')
        if self.reduce_channels.bias is not None:
            init.constant_(self.reduce_channels.bias, 0)

    @staticmethod
    def init_params_(model, pre_trained):
        '''Init layer parameters.'''
        for m in model.modules():
            if pre_trained:
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)