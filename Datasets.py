from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import glob
import os
import fnmatch
import numpy as np
import pdb
import torch
import librosa
import math
import time
import soundfile

AUDIO_EXTENSIONS = ['.wav', '.WAV',]

path_to_coffee_noise = 'coffee_cut_10s_mono_16k.wav'
y_coffee, sr_coffee = soundfile.read(path_to_coffee_noise)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if (os.path.isdir(os.path.join(dir, d)))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def psc_find_classes(classes_file):
    classes = open(classes_file, 'r').readlines()
    classes = [c.strip() for c in classes]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def fixed_augmet_signal(y, noise_type, noise_amp, quiet=True):
    
    y_mod = y

    #p = sp.sum(x*x, 1)/x.size

    p_c = np.mean(abs(y))

    if noise_type == 'gauss':
        noise_sig = np.random.normal(size=len(y))
        p_n = np.mean(abs(noise_sig ))
        y_mod = y_mod +  noise_amp * (float(p_c) / p_n) * noise_sig

    elif noise_type == 'speckle':
        # pdb.set_trace()
        noise_sig = np.random.normal(size=len(y))
        p_n = np.mean(abs(noise_sig * np.std(y)))
        y_mod = y_mod + y_mod * noise_amp * (float(p_c) / p_n) * noise_sig

        # noise_sig = noise_amp * np.random.normal(size=len(y))
        # y_mod = y_mod + y_mod * noise

    elif noise_type == 'coffee':
        y_coffee_temp = y_coffee
        if len(y) < len(y_coffee):
            y_coffee_temp = y_coffee[:len(y)]
        # try:
            # pdb.set_trace()
        p_n = np.mean(abs(y_coffee_temp))
        y_mod = y_mod + noise_amp * (float(p_c) / p_n) * y_coffee_temp
        # except Exception as e:
        #     pdb.set_trace()
        #     print(len(y_coffee_temp))

    # soundfile.write('x.wav', y_mod, 16000)
    # exit()
    return y_mod

class PSCDataset(Dataset):
    def __init__(self, classes_file, data_list, max_audio_len = 10, noise_type = None, noise_amp = 0.0):
        """
        classes_root_dir: the root directory we're not going to use, only need it for the classes (train set)
        this_root_dir: the root directory w're going to use now (train/val/test) - if contains  folders and not file
        :param root_dir:
        :param yolo_config: dictionary that contain the require data for yolo (S, B, C)
        """
        
        classes, class_to_idx = psc_find_classes(classes_file)
        self.class_to_idx = class_to_idx
        self.classes = classes
        self.max_audio_len = max_audio_len
        
        self.noise_type = noise_type
        self.noise_amp = noise_amp

        self.data = data_list
        #self.data = data_list[:1000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        features_path = self.data[idx][0] #wav file
        features, num_of_frames, sr, real_features_len = self.spect_loader(features_path, max_len = 100*self.max_audio_len+1, 
                                                         noise_type = self.noise_type, noise_amp = self.noise_amp)

        max_len_in_frames = self.max_audio_len * sr


        target_path = self.data[idx][1]
        target = open(target_path, "r").readlines()
        #print(target_path)

        num_of_classes = len(self.class_to_idx)
        words_in_wav_file = []
        location_in_wav_file = np.zeros(shape=(num_of_classes, 3))
        for line_str in target:
            #print(line_str)
            split_line = line_str.strip().split("\t")
            start, end, word = split_line[0], split_line[1], split_line[2]
            #print(split_line)

            if int(end) > max_len_in_frames:
                break
            try:
                if not word in self.classes:
                    #print(word)
                    continue
                #print(word)
                object_class = self.class_to_idx[word]
                words_in_wav_file.append(object_class)

                location_in_wav_file[object_class][0] = 1
                location_in_wav_file[object_class][1] = int(start)
                location_in_wav_file[object_class][2] = int(end)

            except:
                print("{} doesn't exist in labels".format(word))
                continue

        target = torch.Tensor([-1]*num_of_classes)
        for class_idx in words_in_wav_file:
            target[class_idx] = 1

        return features, target, location_in_wav_file, features_path

    @staticmethod
    def spect_loader(m_path, window_size=.025, window_stride=.01, n_mfcc = 39, window='hamming', normalize=True, max_len=1000, 
    noise_type = None, noise_amp = 0.0):#1000: 10 secs

        y, sr = soundfile.read(m_path) 

        wav_len = (max_len - 1)/100
        if len(y) > wav_len*sr: 
            y = y[:int(wav_len*sr)]

        if not noise_type == None:
            y = fixed_augmet_signal(y, noise_type, noise_amp)

        try:
            n_fft = int(sr * window_size)
        except:
            print(m_path)
        hop_length = int(sr * window_stride)

        #hop_length=int(0.010sr), n_fft=int(0.025sr)
        #==> set a window width to 25 ms and the stride to 10 ms

        # MFCC
        D = librosa.feature.mfcc(y, n_mfcc = n_mfcc, n_fft=n_fft, hop_length=hop_length, htk = True, n_mels = 40)
        spect = D

        #STFT:
        # D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window)
        # spect, phase = librosa.magphase(D)
        # spect = np.log1p(spect)

        real_features_len = spect.shape[1]
        # make all spects with the same dims
        if spect.shape[1] < max_len:
            pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
            spect = np.hstack((spect, pad))
        elif spect.shape[1] > max_len:
            spect = spect[:, :max_len]

        if spect.shape[0] < n_mfcc:
            pad = np.zeros((n_mfcc - spect.shape[0], spect.shape[1]))
            spect = np.vstack((spect, pad))
        elif spect.shape[0] > n_mfcc:
            spect = spect[:n_mfcc, :]
        spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
        spect = torch.FloatTensor(spect)

        # z-score normalization
        if normalize:
            mean = spect.mean()
            std = spect.std()
            if std != 0:
                spect.add_(-mean)
                spect.div_(std)
        #print('spect time')
        #print(time.time() - start)

        return spect, len(y), sr, real_features_len #length of original file

    def get_filename_by_index(self, idx):
        return self.data[idx][0]

 


class ImbalancedYoloDatasetSampler(Sampler):

    def __init__(self, dataset, indices=None):

        self.dataset = dataset
        if indices != None:
            self.indices = indices
        else:
            self.indices = list(range(len(dataset)))

        self.num_samples = len(self.indices)
        labels_list = []
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(idx)
            labels_list.append(label)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[labels_list[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, idx):
        return self.dataset.get_label(idx)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples




# if __name__ == '__main__':

#     pdb.set_trace()
#     speech_dataset = SpeechDataset(root_dir='/home/mlspeech/segalya/yolo/YoloSpeech2Word/data/mfcc_labels/buckeye', normalize_feats = True)


#     for i in range(len(speech_dataset)):
#         sample = speech_dataset[i]

#         print(sample)