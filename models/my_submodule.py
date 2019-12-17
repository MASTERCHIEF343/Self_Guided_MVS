import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

#dilation空洞卷积增加感受野范围
#padding填充保持输入输出大小相同，pytorch正常是valid卷积
def conv_2d(in_planes, out_planes, kernel_size, stride, pad, dilation):
	return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = dilation if dilation > 1 else pad, \
		dilation = dilation, bias = False),
		nn.BatchNorm2d(out_planes)
	)

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
	return nn.Sequential(
		nn.Conv3d(in_planes, out_planes, kernel_size = kernel_size, padding = pad, stride = stride, bias = False),
		nn.BatchNorm3d(out_planes)
	)

class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Sequential(
			conv_2d(inplanes, planes, 3, stride, pad, dilation),
			nn.LeakyReLU(0.1, inplace = True),
		)
		self.conv2 = conv_2d(planes, planes, 3, 1, pad, dilation)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		if self.downsample is not None:
			x = self.downsample(x)
		out += x
		return out

class disparity_regression(nn.Module):
	def __init__(self, maxdisp):
		super(disparity_regression, self).__init__()
		#[1, 16, 1, 1]
		self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])), requires_grad=False)

	def forward(self, x):
		#使disp和x的size一样
		disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
		out = torch.sum(x * disp, 1)
		return out

class feature_extraction(nn.Module):
	def __init__(self):
		super(feature_extraction, self).__init__()
		self.inplanes = 32
		self.firstconv = nn.Sequential(
			conv_2d(3, 32, 3, 2, 1, 1),
			nn.LeakyReLU(0.1, inplace = True),
			conv_2d(32, 32, 3, 1, 1, 1),
			nn.LeakyReLU(0.1, inplace = True),
			conv_2d(32, 32, 3, 1, 1, 1),
			nn.LeakyReLU(0.1, inplace = True),)

		#residual_block, out:[1/4, 1/4, 128]
		self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
		self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
		self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
		self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

		#spp + interpolation
		self.spp1 = nn.Sequential(
			nn.AvgPool2d((32, 32), stride = (32, 32)),
			conv_2d(128, 32, 1, 1, 0, 1),
			nn.ReLU(inplace = True),
		)

		self.spp2 = nn.Sequential(
			nn.AvgPool2d((16, 16), stride = (16, 16)),
			conv_2d(128, 32, 1, 1, 0, 1),
			nn.ReLU(inplace = True),
		)

		self.spp3 = nn.Sequential(
			nn.AvgPool2d((8, 8), stride = (8, 8)),
			conv_2d(128, 32, 1, 1, 0, 1),
			nn.ReLU(inplace = True),
		)

		self.spp4 = nn.Sequential(
			nn.AvgPool2d((4, 4), stride = (4, 4)),
			conv_2d(128, 32, 1, 1, 0, 1),
			nn.ReLU(inplace = True),
		)

		self.lastconv = nn.Sequential(
			conv_2d(320, 128, 3, 1, 1, 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(128, 32, kernel_size = 1, padding = 0, stride = 1, bias = False))

		self.feature_extraction_test = nn.Sequential(
			conv_2d(3, 32, 7, 1, 1, 1),
			nn.ReLU(inplace = True),
			conv_2d(32, 32, 5, 1, 1, 1),
			nn.ReLU(inplace = True),
			conv_2d(32, 32, 3, 2, 1, 1),
			nn.ReLU(inplace = True),
			conv_2d(32, 32, 3, 1, 1, 1),
			nn.ReLU(inplace = True),
			conv_2d(32, 32, 3, 2, 1, 1),
			nn.ReLU(inplace = True),
			conv_2d(32, 32, 3, 1, 1, 1),
			nn.ReLU(inplace = True),
			conv_2d(32, 32, 3, 1, 1, 1),
			nn.ReLU(inplace = True),
		)

	def _make_layer(self, block, planes, blocks, stride, pad, dilation):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
			nn.BatchNorm2d(planes * block.expansion),)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes,1,None,pad,dilation))
		return nn.Sequential(*layers)


	def forward(self, x):
		#firstconv output: [1/2, 1/2, 32]
		output = self.firstconv(x)
		output = self.layer1(output)
		output_raw = self.layer2(output)
		output = self.layer3(output_raw)
		output_skip = self.layer4(output)
		m = nn.Upsample((output_skip.size()[2], output_skip.size()[3]), mode = 'bilinear')

		output_spp1 = self.spp1(output_skip)
		output_spp1 = m(output_spp1)

		output_spp2 = self.spp2(output_skip)
		output_spp2 = m(output_spp2)

		output_spp3 = self.spp3(output_skip)
		output_spp3 = m(output_spp3)

		output_spp4 = self.spp4(output_skip)
		output_spp4 = m(output_spp4)

		output_feature = torch.cat((output_raw, output_skip, output_spp4, output_spp3, output_spp2, output_spp1), 1)
		output_feature = self.lastconv(output_feature)
		return output_feature
