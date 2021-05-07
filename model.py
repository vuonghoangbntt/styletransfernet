import os
import torch
from PIL import Image, ImageFile
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as f
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from uuid import uuid4

mse_criteration = nn.MSELoss()


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert(len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2)+eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_content_loss(input_im, target):
    assert(input_im.size() == target.size())
    assert(target.requires_grad is False)
    return mse_criteration(input_im, target)


def calc_style_loss(input_im, target):
    assert(input_im.size() == target.size())
    assert(target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input_im)
    target_mean, target_std = calc_mean_std(target)
    return mse_criteration(input_mean, target_mean)+mse_criteration(input_std, target_std)


vggnet = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
              padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),  # relu 1-1
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
              padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                 dilation=1, ceil_mode=False),

    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),
              padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),  # relu 2-1
    # encoder 3-1
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),

    nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                 dilation=1, ceil_mode=False),
    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),  # relu 3-1
    # encoder 4-1
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                 dilation=1, ceil_mode=False),

    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),  # relu 4-1
    # rest of vgg not used
    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                 dilation=1, ceil_mode=False),

    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(
        1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True)
)


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        eps = 1e-5
        mean_x = torch.mean(x, dim=[2, 3])
        mean_y = torch.mean(y, dim=[2, 3])

        std_x = torch.std(x, dim=[2, 3])
        std_y = torch.std(y, dim=[2, 3])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
        mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1)+eps
        std_y = std_y.unsqueeze(-1).unsqueeze(-1)+eps
        out = std_y*((x-mean_x)/std_x)+mean_y
        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class StyleTransferNet(nn.Module):
    def __init__(self, vgg_model, skip_connect=None):
        super().__init__()
        vggnet.load_state_dict(vgg_model)
        # vgg first 21 layers for encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        # Load weight for encoder
        self.encoder.load_state_dict(vggnet[:21].state_dict())
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), padding_mode='reflect'),
        )
        self.skip_weight1 = nn.Parameter(
            torch.randn(1, 1, 256, 256)).to(device)
        self.skip_weight2 = nn.Parameter(
            torch.randn(1, 1, 128, 128)).to(device)
        self.skip_weight3 = nn.Parameter(torch.randn(1, 1, 64, 64)).to(device)
        self.adaIN = AdaIN()
        self.mse_criteration = nn.MSELoss()
        self.skip_connect = skip_connect

    def forward(self, x, alpha=1.0):
        content_img = x[0]
        style_img = x[1]
        encode_content = self.encoder(content_img)
        encode_style = self.encoder(style_img)
        encode_out = self.adaIN(encode_content, encode_style)

        if self.skip_connect == 'content':
            skip1 = self.encoder[:5](content_img)  # 256
            skip2 = self.encoder[5:10](skip1)  # 128
            skip3 = self.encoder[10:19](skip2)  # 64
        elif self.skip_connect == 'style':
            skip1 = self.encoder[:5](style_img)
            skip2 = self.encoder[5:10](skip1)
            skip3 = self.encoder[10:19](skip2)
        elif self.skip_connect == 'normalized_content':
            encoding_content = self.encoder[:5](content_img)
            encoding_style = self.encoder[:5](style_img)
            skip1 = self.adaIN(encoding_content, encoding_style)

            encoding_content = self.encoder[5:10](encoding_content)
            encoding_style = self.encoder[5:10](encoding_style)
            skip2 = self.adaIN(encoding_content, encoding_style)

            encoding_content = self.encoder[10:19](encoding_content)
            encoding_style = self.encoder[10:19](encoding_style)
            skip3 = self.adaIN(encoding_content, encoding_style)

        if self.skip_connect is None:
            gen_img = self.decoder(encode_out)
        else:
            # self.skip_weight1 = nn.Parameter(torch.randn(skip1.shape[1:-1]).unsqueeze(2).unsqueeze(0)).to(device)
            # self.skip_weight2 = nn.Parameter(torch.randn(skip2.shape[1:-1]).unsqueeze(2).unsqueeze(0)).to(device)
            # self.skip_weight3 = nn.Parameter(torch.randn(skip3.shape[1:-1]).unsqueeze(2).unsqueeze(0)).to(device)
            # print(self.skip_weight1.shape)
            gen_img = self.decoder[:3](encode_out) + skip3*self.skip_weight3
            gen_img = self.decoder[3:12](gen_img) + skip2*self.skip_weight2
            gen_img = self.decoder[12:17](gen_img) + skip1*self.skip_weight1
            gen_img = self.decoder[17:](gen_img)

        if self.training:
            encode_gen = self.encoder(gen_img)

            fm11_style = self.encoder[:3](style_img)
            fm11_gen = self.encoder[:3](gen_img)

            fm21_style = self.encoder[3:8](fm11_style)
            fm21_gen = self.encoder[3:8](fm11_gen)

            fm31_style = self.encoder[8:13](fm21_style)
            fm31_gen = self.encoder[8:13](fm21_gen)

            loss_content = self.mse_criteration(encode_out, encode_gen)
            loss_style = self.mse_criteration(torch.mean(fm11_gen, dim=[2, 3]), torch.mean(fm11_style, dim=[2, 3])) +	\
                self.mse_criteration(torch.mean(fm21_gen, dim=[2, 3]), torch.mean(fm21_style, dim=[2, 3])) +	\
                self.mse_criteration(torch.mean(fm31_gen, dim=[2, 3]), torch.mean(fm31_style, dim=[2, 3])) +	\
                self.mse_criteration(torch.mean(encode_gen, dim=[2, 3]), torch.mean(encode_style, dim=[2, 3])) +	\
                self.mse_criteration(torch.std(fm11_gen, dim=[2, 3]), torch.std(fm11_style, dim=[2, 3])) +	\
                self.mse_criteration(torch.std(fm21_gen, dim=[2, 3]), torch.std(fm21_style, dim=[2, 3])) +	\
                self.mse_criteration(torch.std(fm31_gen, dim=[2, 3]), torch.std(fm31_style, dim=[2, 3])) +	\
                self.mse_criteration(
                    torch.std(encode_gen, dim=[2, 3]), torch.std(encode_style, dim=[2, 3]))
            return loss_content, loss_style
        encode_out = alpha*encode_out+(1-alpha)*encode_content
        # gen_img = self.decoder(encode_out)
        return gen_img


def test(input_image, style_image, mode):
    output_format = 'jpg'
    input_image = input_image.convert('RGB')
    style_image = style_image.convert('RGB')
    if mode == 1:
        with torch.no_grad():
            vgg_model = torch.load('vgg_normalized.pth')
            net = StyleTransferNet(vgg_model)
            net.decoder.load_state_dict(torch.load(
                'check_point_epoch_35_10000_samples_v2.pth', map_location=torch.device('cpu'))['net'])
            #net = StyleTransferNet(vgg_model, skip_connect='content')
            # net.decoder.load_state_dict(torch.load('./tensors/check_point_epoch_29_10000_samples.pth')['net'])
            # net.decoder.load_state_dict(torch.load('decoder.pth'))
            net.eval()
            input_image = transforms.Resize(512)(input_image)
            style_image = transforms.Resize(512)(style_image)

            input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)
            style_tensor = transforms.ToTensor()(style_image).unsqueeze(0)

            if torch.cuda.is_available():
                net.cuda()
                input_tensor = input_tensor.cuda()
                style_tensor = style_tensor.cuda()
            out_tensor = net([input_tensor, style_tensor], alpha=1.0)
    else:
        with torch.no_grad():
            vgg_model = torch.load('vgg_normalized.pth')
            net = StyleTransferNet(
                vgg_model, skip_connect='normalized_content')
            net.decoder.load_state_dict(torch.load(
                'check_point_epoch_35_normalized_content_weighted_skip.pth', map_location=torch.device('cpu'))['net'])
            # net.decoder.load_state_dict(torch.load('decoder.pth'))
            net.eval()
            input_image = transforms.Resize((256, 256))(input_image)
            style_image = transforms.Resize((256, 256))(style_image)
            input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)
            style_tensor = transforms.ToTensor()(style_image).unsqueeze(0)
        out_tensor = net([input_tensor, style_tensor], alpha=1.0)
    result_file = uuid4().__str__()[:8]+'-1.jpg'
    save_image(out_tensor, result_file)
    return result_file
