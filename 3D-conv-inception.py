import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
	def __init__(
		self,
		in_channels,
		ch1x1,
		ch3x3red,
		ch3x3,
		ch5x5red,
		ch5x5,
		pooling
		):
		super(Inception, self).__init__()

		# 1x1 conv branch
		self.branch1 = nn.Sequential(
			nn.Conv3d(in_channels, ch1x1, kernel_size=(1, 1, 1), bias=False),
			nn.BatchNorm3d(ch1x1),
			nn.ELU()
			)

		# 1x1 conv + 3x3 conv branch
		self.branch2 = nn.Sequential(
			nn.Conv3d(in_channels, ch3x3red, kernel_size=(1, 1, 1), bias=False),
			nn.BatchNorm3d(ch3x3red),
			nn.ELU(),
			nn.Conv3d(ch3x3red, ch3x3, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False), 
			nn.BatchNorm3d(ch3x3),
			nn.ELU()
			)

		# 1x1 conv + 5x5 conv branch
		self.branch3 = nn.Sequential(
			nn.Conv3d(in_channels, ch5x5red, kernel_size=(1, 1, 1), bias=False),
		 	nn.BatchNorm3d(ch5x5red),
		 	nn.ELU(),
		 	nn.Conv3d(ch5x5red, ch5x5, kernel_size=(5, 5, 5), padding=(2, 2, 2), bias=False), 
		 	nn.BatchNorm3d(ch5x5),
		 	nn.ELU()
		 	)

		# 3x3 pool + 1x1 conv branch
		self.branch4 = nn.Sequential(
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), ceil_mode=True),
		  	nn.Conv3d(in_channels, pooling, kernel_size=(1, 1, 1), bias=False),
	        nn.BatchNorm3d(pooling),
	        nn.ELU()
		  	)

	def forward(self, x):
		branch1 = self.branch1(x)
		#print(branch1.shape)
		branch2 = self.branch2(x)
		#print(branch2.shape)
		branch3 = self.branch3(x)
		#print(branch3.shape)
		branch4 = self.branch4(x)
		#print(branch4.shape)
		return torch.cat([branch1, branch2, branch3, branch4], 1)

class GoogLeNet(nn.Module):
	def __init__(self, num_classes):
		super(GoogLeNet, self).__init__()

		#conv layers before inception
		self.pre_inception = nn.Sequential(
			nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3)),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), ceil_mode=True),
			nn.ELU(),
			nn.Conv3d(64, 64, kernel_size=(1, 1, 1)),
			nn.ELU(),
			nn.Conv3d(64, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), ceil_mode=True),
			nn.ELU()
			)

		self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
		self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
		self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), ceil_mode=True)
		self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
		self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
		self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
		self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
		self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
		self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
		self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.dropout = nn.Dropout(0.4)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, num_classes)

	def forward(self, x):
		x = self.pre_inception(x)
		#print(x.shape)
		x = self.inception3a(x)
		x = self.inception3b(x)
		x = self.maxpool(x)
		x = self.inception4a(x)
		x = self.inception4b(x)
		x = self.inception4c(x)
		x = self.inception4d(x)
		x = self.inception4e(x)
		x = self.maxpool(x)
		x = self.inception5a(x)
		x = self.inception5b(x)
		#print(x.shape)
		x = self.avgpool(x)
		#print(x.shape)
		x = x.view(x.size(0), -1)
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)


model = GoogLeNet(2)
x = torch.randn(2,3, 5,224,224)
y = model(x)
print(y.size())
