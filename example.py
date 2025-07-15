import waven.WaveletGenerator as wg
import waven.Analysis_Utils as au
import waven.LoadPinkNoise as lpn
import waven.zebraGUI as ui
import numpy as np
import gc
import os

"""
Run Pipeline

Parameters Gabor Library:
    N_thetas (int): number of orientatuion equally spaced between 0 and 180 degree.
    Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
    Frequencies (list): spatial frequencies expressed in pixels per cycles.
    Phases (list): 0 and pi/2.
    NX (int): number of azimuth positions (pix) (x shape of the downsampled stimuli).
    NY (int): number of elevation positions (pix) (y shape of the downsampled stimuli).
    Save Path (string): where to save the gabor library

Parameters alignement:
    Dirs (string): where the raw data are.
    Experiment Info: (mouse name, data, experiment number)
    Number of Planes (int): number of acquisition planes.
    Block End (int): timeframe where the experiment starts.
    Number of Frames (int): number of frames stim 30 Hz -> 1800 frame/min.
    Number of Trials to Keep(int): Number of Trials to Keep.

Parameters analysis:
    screen_x: stimulus screen x size inn pixels.
    screen_y: stimulus screen y size inn pixels.
    NX (int): number of azimuth positions (pix) (x shape of the downsampled stimuli).
    NY (int): number of elevation positions (pix) (y shape of the downsampled stimuli).
    Resolution (float): microscope resolution (um per pixels)
    Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
    Visual Coverage (list): [azimuth left, azimuth right, elevation top , elevation bottom] in visual degree.
    Analysis Coverage (list): [azimuth left, azimuth right, elevation top , elevation bottom] in visual degree.
    Movie Path: path to the stimulus (.mp4)
    Library Path: path to Gabor library (same as save path if ran)
    Spks Path (opt): path to the spks.npy file to skip the alignement procedure, if set ignores Parameter alignment

Returns:
    neuron tuning graphs
"""

# Liste des paramètres avec valeurs par défaut
#
gabor_param={
    "N_thetas":"8",
    "Sigmas": "[2, 3, 4, 5, 6, 8]",
    "Frequencies": "[0.015, 0.04, 0.07, 0.1]",
    "Phases": "[0, 90]",
    "NX": "135",
    "NY": "54",
    "Save Path":"/home/sophie/Documents/POSTDOC/TEMP/gabors_library.npy"
}

# Liste des paramètres avec valeurs par défaut
param_defaults = {
    "Path Directory": "/media/sophie/Expansion1/UCL/datatest/videos",
    "Dirs": "/media/sophie/Seagate Basic/datasets",
    "Experiment Info": "('SS002', '2024-07-23', 3)",
    "Number of Planes": "1",
    "Block End": "0",
    "screen_x":"4096",
    "screen_y":"1536",
    "NX": "135",
    "NY": "54",
    "Resolution":"1.3671",
    "Sigmas": "[2, 3, 4, 5, 6, 8]",
    "Frequencies": "[0.015, 0.04, 0.07, 0.1]",
    "Visual Coverage":"[-135, 45, 34, -34]",
    "Analysis Coverage": "[-135, 0, 34, -34]",
    "Number of Frames": "18000",
    "Number of Trials to Keep": "3",
    "Movie Path": "/home/sophie/Documents/POSTDOC/TEMP/videos/perlin_stimulus_10min.mp4",
    "Library Path": "/home/sophie/Documents/POSTDOC/TEMP/gabors_library.npy",
    "Spks Path": "None"
}


sigmas = eval(gabor_param["Sigmas"])
nx = int(gabor_param["NX"])
ny = int(gabor_param["NY"])
n_theta = int(gabor_param["N_thetas"])
offsets= eval(gabor_param["Phases"])
path_save = gabor_param["Save Path"]
xs = np.arange(nx)
ys = np.arange(ny)
thetas = np.array([(i * np.pi) / n_theta for i in range(n_theta)])
sigmas = np.array(sigmas)
offsets=np.array(offsets)
f =  eval(gabor_param["Frequencies"])

path_directory = param_defaults["Path Directory"]
dirs = [param_defaults["Dirs"]]
exp_info = eval(param_defaults["Experiment Info"])
sigmas = eval(param_defaults["Sigmas"])
sigmas=np.array(sigmas)
visual_coverage = eval(param_defaults["Visual Coverage"])
analysis_coverage = eval(param_defaults["Analysis Coverage"])
n_planes = int(param_defaults["Number of Planes"])
block_end = int(param_defaults["Block End"])
screen_x = int(param_defaults["screen_x"])
screen_y = int(param_defaults["screen_y"])
ns = len(sigmas)
resolution=float(param_defaults["Resolution"])
spks_path = param_defaults["Spks Path"]
nb_frames = int(param_defaults["Number of Frames"])
n_trial2keep = int(param_defaults["Number of Trials to Keep"])
movpath = param_defaults["Movie Path"]
lib_path = param_defaults["Library Path"]
screen_ratio = abs(visual_coverage[0]-visual_coverage[1])/nx
xM, xm, yM, ym = analysis_coverage

pathdata = os.path.join(os.path.join(os.path.join(dirs[0] , exp_info[0]) , exp_info[1]) , str(exp_info[2]))
pathsuite2p = os.path.join(pathdata , 'suite2p')

deg_per_pix=abs(xM-xm)/nx
sigmas_deg=np.trunc(2*deg_per_pix*sigmas*100)/100






## if prefer UI
ui.run(param_defaults,gabor_param)







## create a new gabor library
if f!=0:
    freq=True # frquencies and size will be decoupled
    L = wg.makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, f)
else:
    freq=False # frquencies and size will be linearly related
    L = wg.makeFilterLibrary(xs, ys, thetas, sigmas, offsets, f, freq=freq)

np.save(path_save, L)
lib_path=path_save






## define visual coverage for the analysis
if (visual_coverage!=analysis_coverage):
    visual_coverage=np.array(visual_coverage)
    analysis_coverage=np.array(analysis_coverage)
    ratio_x=1-((visual_coverage[0]-visual_coverage[1])-(analysis_coverage[0]-analysis_coverage[1]))/(visual_coverage[0]-visual_coverage[1])
    ratio_y=1-((visual_coverage[2]-visual_coverage[3])-(analysis_coverage[2]-analysis_coverage[3]))/(visual_coverage[2]-visual_coverage[3])
else:
    ratio_x=1
    ratio_y=1





## downsamples and wavelet transforms the stimulus
wg.downsample_video_binary(movpath,visual_coverage,  analysis_coverage, shape=(ny, nx), chunk_size=1000, ratios=(ratio_x, ratio_y))
path=os.path.dirname(movpath)
videodata=np.load(movpath[:-4]+'_downsampled.npy')
videodata=videodata.astype(int)-np.logical_not(videodata).astype(int) # makes the black value -1 and white 1 intead of [0, 1]

wg.waveletDecomposition(videodata, 0, sigmas, path, lib_path)
wg.waveletDecomposition(videodata, 1, sigmas, path, lib_path)







## run data alignment if the neural and stimulis data are acquired with CortexLab system
spks, spks_z, neuron_pos=lpn.loadSPKMesoscope(exp_info, dirs, pathsuite2p, block_end, n_planes, nb_frames, threshold=1.25, last=True,  method='frame2ttl')

## otherwise
spks=np.load(spks_path)
parent_dir = os.path.dirname(spks_path)
neuron_pos=np.load(os.join(parent_dir, 'pos.npy'))

## converts neuron position in microns
neuron_pos=lpn.correctNeuronPos(neuron_pos, resolution)









## the spikes data have to be time registered to the stimulus frames
respcorr_zebra = au.repetability_trial3(spks, neuron_pos, plotting=True)
wavelets0, wavelets1, wavelet_c = lpn.coarseWavelet(path,False, nx, ny, 27, 11, n_theta, ns)

## runs correlation analysis
rfs_zebra = au.PearsonCorrelationPinkNoise(wavelet_c.reshape(18000, -1), np.mean(spks[:, :18000], axis=0),
                                           neuron_pos, 27, 11, ns, analysis_coverage, screen_ratio, sigmas_deg,
                                           plotting=True)
## plot neuron receptive field
idx=2441
au.Plot_RF(rfs_zebra[0][idx],4, title=np.max(rfs_zebra[0][idx]))

## plots neuron tuning curves
tuning_curve=au.PlotTuningCurve(rfs_zebra, 2441, analysis_coverage, sigmas_deg, screen_ratio)


## run Model on all neurons at once and save the result
# simple version (fast)
Predictions, Params, nonlinParams,RhoPhiParams,Metrics, OS=run_Model(rfs, np.array(maxes1), np.array(maxes0), spks, wavelets1, wavelet0, double_wavelet_model=False)

# high granularity version (full model, slower)
# if low RAM, set memmaping=True, if RAM >=120 GB you can set memmmapping=False, il will be faster
results=run_Full_Model( maxes1, maxes0, spks,idxs,thetas,sigmas, frequencies,visual_coverage, neuron_pos,
                    wavelet_path='/media/sophie/Expansion1/UCL/utils/2screens/10/',
                    savepath = '/home/sophie/Pictures/img zebra/supp/supp/', n_min=5, tt=[10, 18000],
                    memmapping=True, train_idx=[0, 2],test_idx=[1, 3],double_wavelet_model=False, lastmin=False,
                    plotting=False )



