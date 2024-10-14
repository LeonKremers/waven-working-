import zebrAnalysis3.WaveletGenerator as wg
import zebrAnalysis3.Analysis_Utils as au
import zebrAnalysis3.LoadPinkNoise as lpn
import numpy as np
import gc

## downsamples and wavelet transforms the stimulus
wg.downsample_video_binary('/home/sophie/Projects/pachaging/ressources/pink_noise_3_5min.mp4')
videodata=np.load('/home/sophie/Projects/pachaging/ressources/pink_noise_3_5min._downsampled.npy')
path='/home/sophie/Projects/pachaging/ressources/'
wg.waveletDecomposition(videodata, 0, path)
wg.waveletDecomposition(videodata, 1, path)

## parameter for the recorded neural datas
pathdir='/media/sophie/Seagate Basic/video/2screens/10/'
dirs = ['/media/sophie/Seagate Basic/datasets']
exp_info = ('SS002','2024-08-21', 1)
n_planes=1
block_end=9009
nx=135
ny=54
nb_frames=18000
n_trial2keep=3
path='/media/sophie/Expansion1/UCL/datatest/videos/2screens/10/'#'/media/sophie/Seagate Basic/video/2screens/10/'
pathdata=dirs[0]+'/'+exp_info[0]+'/'+exp_info[1]+'/'
pathsuite2p=pathdata+'/suite2p'
downsampling=False


## if the neural and stimulis data are acquired with CortexLab system
spks, neuron_pos=lpn.loadSPKMesoscope(exp_info, dirs, pathsuite2p, block_end, n_planes, nb_frames, first=True,  method='photosensor')
neuron_pos=lpn.correctNeuronPos(neuron_pos)

## the spikes data have to be time registered to the stimulus frames
respcorr_zebra = au.repetability_trial3(spks, neuron_pos)
wavelets0, wavelets1, wavelet_c = lpn.coarseWavelet(path, downsampling)
rfs_zebra =  au.PearsonCorrelationPinkNoise(wavelet_c.reshape(18000, -1), np.mean(spks[:, :18000], axis=0),
                                  neuron_pos, 27, 11, plotting=True)