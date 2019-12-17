import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from models.my_submodule import *
from models.submodule import convbn
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
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace = True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace = True),
        )

        self.max1 = nn.MaxPool3d((2, 1, 1), return_indices=True)

        self.max1_1 = nn.Sequential(
            nn.Conv3d(32, 32, 1, 1, 0, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.max2 = nn.MaxPool3d((2, 1, 1), return_indices=True)

        self.max2_1 = nn.Sequential(
            nn.Conv3d(32, 32, 1, 1, 0, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.max3 = nn.MaxPool3d((2, 1, 1), return_indices=True)

        self.max3_1 = nn.Sequential(
            nn.Conv3d(32, 32, 1, 1, 0, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.max4 = nn.MaxPool3d((2, 1, 1), return_indices=True)

        self.max4_1 = nn.Sequential(
            nn.Conv3d(32, 32, 1, 1, 0, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.unpool1 = nn.MaxUnpool3d((2, 1, 1), stride=(2, 1, 1))
        self.unpool1_1 = nn.Sequential(
		    nn.Conv3d(32, 32, 1, 1, 0, 1),
		    nn.BatchNorm3d(32)
		)

        self.unpool2 = nn.MaxUnpool3d((2, 1, 1), stride=(2, 1, 1))
        self.unpool2_1 = nn.Sequential(
		    nn.Conv3d(32, 32, 1, 1, 0, 1),
		    nn.BatchNorm3d(32)
		)

        self.unpool3 = nn.MaxUnpool3d((2, 1, 1), stride=(2, 1, 1))
        self.unpool3_1 = nn.Sequential(
		    nn.Conv3d(32, 32, 1, 1, 0, 1),
		    nn.BatchNorm3d(32)
		)

        self.unpool4 = nn.MaxUnpool3d((2, 1, 1), stride=(2, 1, 1))
        self.unpool4_1 = nn.Sequential(
            nn.Conv3d(32, 32, 1, 1, 0, 1),
	    	nn.BatchNorm3d(32),
        )

        self.todepth = nn.Sequential(
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
            cost0, indices1 = self.max1(cost0)
            cost0 = self.max1_1(cost0)
            cost0, indices2 = self.max2(cost0)
            cost0 = self.max2_1(cost0)
            cost0, indices3 = self.max3(cost0)
            cost0 = self.max3_1(cost0)
            cost0, indices4 = self.max4(cost0)
            cost0 = self.max4_1(cost4)
            cost0 = self.unpool4(cost0, indices4)
            cost0 = self.unpool4_1(cost0)
            cost0 = self.unpool3(cost0, indices3)
            cost0 = self.unpool3_1(cost0)
            cost0 = self.unpool2(cost0, indices2)
            cost0 = self.unpool2_1(cost0)
            cost0 = self.unpool1(cost0, indices1)
            cost0 = self.unpool1_1(cost0)
            cost0 = self.todepth(cost0)

            if j == 0:
                cost_volume = cost0
            else:
                cost_volume += cost0
        cost_volume = cost_volume / len(refs)
        cost_volume = F.upsample(cost_volume, [self.nlabel,target_img.size()[2],target_img.size()[3]], mode='trilinear')
        cost_volume = torch.squeeze(cost_volume, 1)
        pred0 = F.softmax(cost_volume, dim = 1)
        pred0 = disparity_regression(self.nlabel)(pred0)
        depth0 = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)
        return depth0
