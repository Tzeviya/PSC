import torch.optim as optim
import torch
import argparse
import psc_model
from train_psc import train, test
import numpy as np
import glob
import time
import os
import pdb
import Datasets


parser = argparse.ArgumentParser(description='train classification for yolo model')
parser.add_argument('--train_data', type=str, default='train_data_list.txt',
                    help='location of the train data list')
parser.add_argument('--val_data', type=str, default='dev_data_list.txt',
                    help='location of the validation data list')
parser.add_argument('--classes_file', type=str, default='words_for_classification_1000_libri_960.txt', help='file with 1000 words to classify')
parser.add_argument('--opt', type=str, default='sgd', help='optimization method')
parser.add_argument('--momentum', type=float, default='0.9', help='momentum')
parser.add_argument('--prev_class_model', type=str, default='', help='the location of the prev classification model')
parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=2000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float,  default=0.0, help='dropout probability value')
parser.add_argument('--seed', type=int, default=1245, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--patience', type=int, default=10, metavar='N', 
                    help='how many epochs of no loss improvement should we wait before stop training')
parser.add_argument('--psc_type', type=str, default='big',  help='small || big')
parser.add_argument('--dilation', action='store_true', help='use dilation (DONT USE THIS OPTION')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',  help='report interval')
parser.add_argument('--save_folder', type=str,  default='psc_models', help='path to save the final model')
parser.add_argument('--save_file', type=str,  default='x.pth', help='filename to save the final model')
parser.add_argument('--trained_model', type=str, default='', help='load model already trained by this script')
parser.add_argument('--audio_len', type=int, default=10, help='cut audio file to this length')
parser.add_argument('--noise_amp', type=float, default=0.0, help='how loud the background noise should be')
parser.add_argument('--noise_type', type=str, default=None, help='None || gauss || speckle || coffee ')



args = parser.parse_args()
print(args)

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# build model
# if args.prev_class_model: #pretrained by other model
#     model_ = PSC(kernel_sizes = [9, 10], num_MFCC_coeffs = 39, len_last_layer = 1000, train_mode = True)
#     checkpoint = torch.load(args.prev_class_model, map_location=lambda storage, loc: storage) #will forcefully remap everything onto CPU
#     speech_net.load_state_dict(checkpoint['net'])
# else: #initialize

if args.psc_type == 'small':
    speech_net = psc_model.PSC6(kernel_sizes = [9, 10], num_MFCC_coeffs = 39, len_last_layer = 1000, dilation = args.dilation, train_mode = True)
else:
    speech_net = psc_model.PSC10(kernel_sizes = [5, 10], num_MFCC_coeffs = 39, len_last_layer = 1000, dilation = args.dilation, train_mode = True)
    
#pdb.set_trace()
if os.path.isfile(args.trained_model): #model exists  
    speech_net = psc_model.load_model(args.trained_model)
    
if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    speech_net = torch.nn.DataParallel(speech_net).cuda()
    # model = speech_net.cuda()

# define optimizer
if args.opt.lower() == 'adam':
    optimizer = optim.Adam(speech_net.parameters(), lr=args.lr)
elif args.opt.lower() == 'sgd':
    optimizer = optim.SGD(speech_net.parameters(), lr=args.lr, momentum=args.momentum)
else:
    optimizer = optim.SGD(speech_net.parameters(), lr=args.lr, momentum=args.momentum)

f_train = np.loadtxt(args.train_data, dtype='str', delimiter='\t')
f_train = f_train.tolist()
train_dataset = Datasets.PSCDataset(classes_file = args.classes_file, data_list = f_train, max_audio_len = args.audio_len,
                                    noise_type = args.noise_type, noise_amp = args.noise_amp)

f_val = np.loadtxt(args.val_data, dtype='str', delimiter='\t')
f_val = f_val.tolist()
val_dataset = Datasets.PSCDataset(classes_file = args.classes_file, data_list = f_val, max_audio_len = args.audio_len,
                                  noise_type = args.noise_type, noise_amp = args.noise_amp)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, 
    num_workers=10, pin_memory=args.cuda, sampler = None)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=10, pin_memory=args.cuda, sampler=None)

best_valid_loss = np.inf
iteration = 0
epoch = 1


# trainint with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience):

    train(train_loader, speech_net, 1.0, optimizer, epoch, args.cuda, args.log_interval)
    valid_loss = test(val_loader, speech_net, 1.0, args.cuda)
    if valid_loss >= best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        print('Saving model...')
        iteration = 0
        best_valid_loss = valid_loss
        state = {
            'net': speech_net.module.state_dict() if args.cuda else speech_net.state_dict(),
            'acc': valid_loss,
            'epoch': epoch,
            'params': {'dilation': args.dilation, 'size': args.psc_type}
        }
        if not os.path.isdir(args.save_folder):
            os.mkdir(args.save_folder)
        torch.save(state, args.save_folder +'/' + args.save_file)
    epoch += 1

