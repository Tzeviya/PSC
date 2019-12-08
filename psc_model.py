import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

class PSC6(nn.Module): #Visually grounded learning of keyword prediction from untranscribed speech
	'''
	Kemper Conv Model:
	"filter_shapes": [
		[39, 9, 1, 96], 
		[1, 10, 96, 96],
		[1, 10, 96, 96],
		[1, 10, 96, 96],
		[1, 10, 96, 96],
		[1, 10, 96, 1000],

	'''
	def __init__(self, kernel_sizes = [9, 10], num_MFCC_coeffs = 39, len_last_layer = 1000, dilation = False, train_mode = True):
		super(PSC6, self).__init__()
		self.train_mode = train_mode     
		#used 2d because maybe 1d can't cope with batch dim
		#padding: can't keep dim with kernel 10. x axis sizes: 101, 101, 100, 99, 98, 97, 96

		if dilation == False:
			self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = (num_MFCC_coeffs, kernel_sizes[0]), padding=(0, 4)) #kernel is num feats x kernel size
			self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (1, kernel_sizes[1]), padding=(0, 4))
			self.conv3 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (1, kernel_sizes[1]), padding=(0, 5))
			self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (1, kernel_sizes[1]), padding=(0, 4))
			self.conv5 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (1, kernel_sizes[1]), padding=(0, 5))
			self.conv6 = nn.Conv2d(in_channels = 96, out_channels = len_last_layer, kernel_size = (1, kernel_sizes[1]), padding=(0, 4))

		else:
			self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = (num_MFCC_coeffs, kernel_sizes[0]), padding=(0, 4)) #kernel is num feats x kernel size
			self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv3 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv5 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv6 = nn.Conv2d(in_channels = 96, out_channels = len_last_layer, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))

		self.init_weight()

	def forward(self, x):
		x = F.relu(self.conv1(x)) #relu convolution
		x = F.relu(self.conv2(x)) #relu convolution
		x = F.relu(self.conv3(x)) #relu convolution
		x = F.relu(self.conv4(x)) #relu convolution
		x = F.relu(self.conv5(x)) #relu convolution
		x = self.conv6(x) #linear convolution

		return x

	def init_weight(self):
		nn.init.xavier_normal_(self.conv1.weight)
		nn.init.xavier_normal_(self.conv2.weight)
		nn.init.xavier_normal_(self.conv3.weight)
		nn.init.xavier_normal_(self.conv4.weight)
		nn.init.xavier_normal_(self.conv5.weight)
		nn.init.xavier_normal_(self.conv6.weight)
		

#=======================================================================================

class PSC10(nn.Module): #Jointly Learning to Locate and Classify Words using Convolutional Networks

	def __init__(self, kernel_sizes = [5, 10], num_MFCC_coeffs = 39, len_last_layer = 1000, dilation = False, train_mode = True):
		super(PSC10, self).__init__()
		self.train_mode = train_mode     
		#used 2d because maybe 1d can't cope with batch dim
		#padding: can't keep dim with kernel 10. x axis sizes: 101, 101, 100, 99, 98, 97, 96.... 93

		if dilation == False:
			self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 80, kernel_size = (num_MFCC_coeffs, kernel_sizes[0]), padding=(0, 2)) #kernel is num feats x kernel size
			self.conv2 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 5))
			self.conv3 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 4))
			self.conv4 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 5))
			self.conv5 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 4))
			self.conv6 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 5))
			self.conv7 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 4))
			self.conv8 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 5))
			self.conv9 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 4))
			self.conv10 = nn.Conv2d(in_channels = 80, out_channels = len_last_layer, kernel_size = (1, kernel_sizes[1]), padding=(0, 4))

		else:#trying dilation:
			self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 80, kernel_size = (num_MFCC_coeffs, kernel_sizes[0]), padding=(0, 2)) #kernel is num feats x kernel size
			self.conv2 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv3 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv4 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv5 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv6 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv7 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv8 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv9 = nn.Conv2d(in_channels = 80, out_channels = 80, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
			self.conv10 = nn.Conv2d(in_channels = 80, out_channels = len_last_layer, kernel_size = (1, kernel_sizes[1]), padding=(0, 9), dilation = (1,2))
		
		self.init_weight()

	def forward(self, x):
		#pdb.set_trace()
		x = F.relu(self.conv1(x)) #relu convolution
		x = F.relu(self.conv2(x)) #relu convolution
		x = F.relu(self.conv3(x)) #relu convolution
		x = F.relu(self.conv4(x)) #relu convolution
		x = F.relu(self.conv5(x)) #relu convolution
		x = F.relu(self.conv6(x)) #relu convolution
		x = F.relu(self.conv7(x)) #relu convolution
		x = F.relu(self.conv8(x)) #relu convolution
		x = F.relu(self.conv9(x)) #relu convolution
		x = self.conv10(x) #linear convolution

		return x

	def init_weight(self):
		nn.init.xavier_normal_(self.conv1.weight)
		nn.init.xavier_normal_(self.conv2.weight)
		nn.init.xavier_normal_(self.conv3.weight)
		nn.init.xavier_normal_(self.conv4.weight)
		nn.init.xavier_normal_(self.conv5.weight)
		nn.init.xavier_normal_(self.conv6.weight)
		nn.init.xavier_normal_(self.conv7.weight)
		nn.init.xavier_normal_(self.conv8.weight)
		nn.init.xavier_normal_(self.conv9.weight)
		nn.init.xavier_normal_(self.conv10.weight)

def load_model(save_dir):

	checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
	dilation, size = checkpoint['params']['dilation'], checkpoint['params']['size']

	if size == 'big':
		speech_net = PSC10(train_mode = False, dilation = dilation)
	else:
		speech_net = PSC6(train_mode = False, dilation = dilation)

	#speech_net = PSC6(train_mode = False, dilation = False)
	speech_net.load_state_dict(checkpoint['net'])

	return speech_net