import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from models.my_submodule import *
from inverse_warp import inverse_warp
import math
import torch.nn.functional as F

class YunNet(nn.Module):
	def __init__(self, nlabel, mindepth):
		super(YunNet, self).__init__()
		self.nlabel = nlabel
		self.mindepth = mindepth

		#spp
		self.feature_extraction = feature_extraction()

		#3DCNN
		self.conv3d0 = nn.Sequential(
			convbn_3d(65, 32, 3, 1, 1),
			nn.ReLU(inplace = True),
			convbn_3d(32, 32, 3, 1, 1),
			nn.ReLU(inplace = True),
		)

		self.conv3d0_2 = nn.Sequential(
			convbn_3d(32, 32, 1, 1, 0),
			nn.ReLU(inplace = True),
		)

		self.conv3d1 = nn.Sequential(
			convbn_3d(32, 64, 3, 2, 1),
			nn.ReLU(inplace = True),
			convbn_3d(64, 64, 3, 1, 1),
		)

		self.conv3d1_2 = nn.Sequential(
			convbn_3d(64, 64, 1, 1, 1),
			nn.ReLU(inplace = True),
		)

		self.conv3d2 = nn.Sequential(
			convbn_3d(64, 128, 3, 2, 1),
			nn.ReLU(inplace = True),
			convbn_3d(128, 128, 3, 1, 1),
		)

		self.conv3d2_2 = nn.Sequential(
			convbn_3d(128, 128, 1, 1, 0),
			nn.ReLU(inplace = True),
		)

		self.conv2_3 = nn.Sequential(
			nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
			nn.BatchNorm3d(64)
		)

		self.conv3_2 = nn.Sequential(
			nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
			nn.BatchNorm3d(32)
		)

		self.cost_disp1 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode = 'trilinear'),
			nn.ReLU(inplace = True),
			convbn_3d(64, 1, 3, 1, 1),
		)

		self.cost_disp2 = nn.Sequential(
			convbn_3d(32, 32, 3, 1, 1),
			nn.ReLU(inplace = True),
			convbn_3d(32, 1, 3, 1, 1),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.xavier_uniform(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, target_img, refs, pose, intrinsics, intrinsics_inv):
		#内参
		intrinsics4 = intrinsics.clone()
		intrinsics_inv4 = intrinsics_inv.clone()
		intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
		intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4

		target_fea = self.feature_extraction(target_img)
		# print("featrue extracture shape: {}".format(target_fea.shape))
		disp2depth = Variable(torch.ones(target_fea.size(0), target_fea.size(2), target_fea.size(3))).cuda() * self.mindepth * self.nlabel
		sum_tensor = Variable(torch.ones(target_fea.size(0), 1, target_fea.size(2), target_fea.size(3))).cuda()
		for j, ref in enumerate(refs):
			#[Batch_size, 64channel(ref + target), 16, 1/4 H, 1/4 W]
			cost = Variable(torch.FloatTensor(target_fea.size()[0], target_fea.size()[1] * 2, self.nlabel, target_fea.size()[2], target_fea.size()[3]).zero_()).cuda()

			ref_fea = self.feature_extraction(ref)
			for i in range(self.nlabel):
				depth = torch.div(disp2depth, (i+1e-16))
				ref_fea_t = inverse_warp(ref_fea, depth, pose[:,j], intrinsics4, intrinsics_inv4)

				cost[:, :target_fea.size()[1], i, :,:] = target_fea
				cost[:, target_fea.size()[1]:, i, :,:] = ref_fea_t

			cost = cost.contiguous()
			cost0 = self.conv3d0(cost)
			cost0_2 = self.conv3d0_2(cost0)

			cost1 = self.conv3d1(cost0)
			cost1_2 = self.conv3d1_2(cost1)

			cost2 = self.conv3d2(cost1)
			cost2_2 = self.conv3d2_2(cost2)
			cost2_3 = self.conv2_3(cost2_2)

			cost3 = cost1_2 + cost2_3

			cost4 = self.conv3_2(cost3)

			cost4 = cost4 + cost0_2
			cost_disp2 = self.cost_disp2(cost4)

			if j == 0:
				cost_volume = cost_disp2
			else:
				cost_volume += cost_disp2
		cost_volume = cost_volume / len(refs)
		cost_volume = F.upsample(cost_volume, [self.nlabel,target_img.size()[2],target_img.size()[3]], mode='trilinear')
		cost_volume = torch.squeeze(cost_volume, 1)
		# print(cost_volume.shape)
		pred0 = F.softmax(cost_volume, dim = 1)
		pred0 = disparity_regression(self.nlabel)(pred0)
		# print("pred0: {}".format(pred0[0]))
		depth0 = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)
		# depth0 = 50 / (pred0.unsqueeze(1)+1e-16)
		# print("depth0: {}".format(depth0[0]))

		return depth0

if __name__ == '__main__':
	dnet = DNet()
	test = torch.randn([8, 1, 240, 320])
	test = Variable(dnet(test))
	label = Variable(torch.full((test.size()), 1))
	print(label.shape)
	criterion = torch.nn.BCELoss()
	res = criterion(label, test)
	print(res)
