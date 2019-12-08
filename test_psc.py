import torch.optim as optim
import torch
import argparse
import numpy as np
import time
import os
import pdb
import Datasets
import psc_model
import train_psc 

parser = argparse.ArgumentParser(description='test classification for psc model')
parser.add_argument('--test_data', type=str, default='test_data_list.txt',
                    help='location of the validation data')
parser.add_argument('--classes_file', type=str, default='words_for_classification_1000_libri_960.txt', help='file with 1000 words to classify')

parser.add_argument('--model', type=str, default='psc_models/psc_model_conv10_march17.pth',
                    help='the location of the prev classification model')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1245,
                    help='random seed')
parser.add_argument('--psc_type', type=str, default='big',  help='small || big')
parser.add_argument('--theta_range', type=str, default='0.1_1.0_0.1',  help='0.1_1.0_0.1')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--wav_len', type=int, default=10, help='len of audio in seconds')
parser.add_argument('--noise_amp', type=float, default=0.0, help='how loud the background noise should be')
parser.add_argument('--noise_type', type=str, default=None, help='None || gauss || speckle || coffee ')


args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


model = psc_model.load_model(args.model)
if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

f_test = np.loadtxt(args.test_data, dtype='str', delimiter='\t')
f_test = f_test.tolist()
test_dataset = Datasets.PSCDataset(classes_file=args.classes_file, data_list=f_test, max_audio_len = args.wav_len,
                                    noise_type = args.noise_type, noise_amp = args.noise_amp)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=args.cuda, sampler=None)

range_split = args.theta_range.split('_')
start_theta = float(range_split[0])
end_theta = float(range_split[1])
step_theta = float(range_split[2])

atwv_dict = {} #dict: theta: atwv val
for theta in np.arange(start_theta, end_theta, step_theta):
    acc = train_psc.test_acc(test_loader, model, args.wav_len, 1.0, theta, args.cuda)
    atwv = train_psc.keyword_spotting_test_mtwv(test_loader, model, args.wav_len, 1.0, theta, args.cuda)
    atwv_dict[theta] = atwv

for key in sorted(atwv_dict):
    print(key, atwv_dict[key])