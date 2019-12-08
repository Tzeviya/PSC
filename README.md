# PSC

PSC CNN, after **P**alaz, **Sy**nnaeve, and **C**ollobert.

Python implementation for ["Jointly Learning to Locate and Classify Words using Convolutional Networks"](https://ronan.collobert.com/pub/matos/2016_wordaggr_interspeech.pdf).

Also implemented the version described in ["Visually grounded learning of keyword prediction from untranscribed speech"](https://arxiv.org/pdf/1703.08136.pdf).


- Data:

The librispeech dataset can be found [here](http://www.openslr.org/12). The data is already divided to train, validation and test sets. Both the `clean` and `other` segments have been used for training, but were tested separately.

The dataset is given with `.wav` and `.lab` files, where the `.lab` files are transcriptions. In order to obtain word alignemnts, the Montreal Forced Aligner ([MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/)) can be used.

Audio files are given as `.wav` files. Convert their transcriptions into `.wrd` files in the following format:

````
2720	11680	chapter	
13920	24320	eleven	
95520	96960	the	
96960	104160	morrow	
104160	108160	brought	
108160	108640	a	
108640	113280	very	
113280	119680	sober	
119680	125600	looking	
125600	133120	morning	
````

That is, every line contains (start, end , word), which denote the start and end times (in frames) for a given word. 

Create 3 text files: `train_data_list.txt`, `dev_data_list.txt`, and `test_data_list.txt`, each containing a list of `.wav`s and their corresponding `.wrd` files, for example:

````
/librispeech_Dev/LibriSpeech_cnvt/7850/281318/7850-281318-0010.wav	/librispeech_Dev/LibriSpeech_wrd/7850/281318/7850-281318-0010.wrd
/librispeech_Dev/LibriSpeech_cnvt/7850/281318/7850-281318-0009.wav	/librispeech_Dev/LibriSpeech_wrd/7850/281318/7850-281318-0009.wrd
/librispeech_Dev/LibriSpeech_cnvt/7850/281318/7850-281318-0011.wav	/librispeech_Dev/LibriSpeech_wrd/7850/281318/7850-281318-0011.wrd
/librispeech_Dev/LibriSpeech_cnvt/7850/281318/7850-281318-0003.wav	/librispeech_Dev/LibriSpeech_wrd/7850/281318/7850-281318-0003.wrd
````

- Train:

Run:

````
python run_psc.py --train_data [path to train_data_list.txt]  
				  --val_data [path to dev_data_list.txt] 
				  --psc_type [choose "big" or "small"]
				  --save_file [name of model file]
				  --cuda 
	                       
````

For `psc_type` , `big` denotes the network structure described in "Jointly Learning to Locate and Classify Words using Convolutional Networks", and `small` denotes the network structure described in "Visually grounded learning of keyword prediction from untranscribed speech".