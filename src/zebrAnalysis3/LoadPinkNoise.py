from .suite2p.utils import cortex_lab_utils as clu
from .suite2p.utils import timelinepy as tlu
from .suite2p.utils import twophopy as tpu
from .suite2p.utils import mpeppy as mpu
from .suite2p.utils import retinotopy as rt
from .suite2p.utils import utils as utils
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as sio
import skimage
from skimage import transform
import os
import matplotlib
matplotlib.use('TkAgg')
from skimage.measure import block_reduce
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
import gc
from sklearn.linear_model import Lasso
import random
from scipy import stats
from scipy.sparse.linalg import svds
import scipy.optimize as opt
import torch
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from . Analysis_Utils import *


def load_wavelets(pathdir, nx, ny, wavelets_r, wavelets_i, direction=False):
    w_r = wavelets_r.reshape((-1, ny, nx, 3, 9))
    w_i = wavelets_i.reshape((-1, ny, nx, 3, 9))
    cos = np.clip(w_r, a_min=0, a_max=None)
    cos_m = np.clip(-w_r, a_min=0, a_max=None)
    sin = np.clip(w_i, a_min=0, a_max=None)
    sin_m = np.clip(-w_i, a_min=0, a_max=None)
    sigma = 1
    pn_wavelets = cos + cos_m + sin + sin_m
    pn_wavelets = pn_wavelets.reshape((-1, ny, nx, 27))
    pn_wavelets = np.array(
        [pn_wavelets[i] / (sigma + np.sum(pn_wavelets[i], axis=0)) for i in range(pn_wavelets.shape[0])])
    pn_wavelets = np.reshape(pn_wavelets, (-1,  ny, nx, 3, 9))
    print(pn_wavelets.shape)
    del cos, cos_m, sin, sin_m, w_r, w_i
    gc.collect()

    if direction:
        sh=np.array(wavelets_i.shape)
        sh[0]=sh[0]+1
        phase = np.insert(np.arctan(wavelets_i / wavelets_r), [0],  np.zeros((ny, nx, 3, 9)), 0).reshape(sh)
        phase_diff = np.diff(phase, axis=0)
        # print(phase_diff.shape)
        # pn_wavelets_fwd=pn_wavelets * np.clip(phase_diff, 0, None)
        # pn_wavelets_bckwd = pn_wavelets * np.clip(-phase_diff, 0, None)
        # print(pn_wavelets_fwd.shape)
        # return pn_wavelets, pn_wavelets_fwd, pn_wavelets_bckwd
        return pn_wavelets, phase_diff
    # pn_wavelets = skimage.transform.resize(pn_wavelets, (pn_wavelets.shape[0], 8, 30, 3, 9))
    else:
        return pn_wavelets


def load_stimulus(pathdir, wavelets_r, wavelets_i, nx=161, ny=60):
    wavelets_r = np.load(pathdir + '/cwt_pn_real_1_9000.npy')
    wavelets_i = np.load(pathdir + '/cwt_pn_imag_1_9000.npy')
    ## load wavelets
    scale = 2
    direct = True
    if not direct:
        wavelets = load_wavelets(pathdir, nx, ny, direction=direct)
        wavelets = wavelets[:, :, :, scale, :]  # only select the higheest scale
        wavelets = skimage.transform.resize(wavelets, (wavelets.shape[0], 8, 20, 9))
        wavelets = wavelets.reshape((wavelets.shape[0], wavelets.shape[1] * wavelets.shape[2] * wavelets.shape[3]))
    elif direct:
        w_r = skimage.transform.resize(wavelets_r.reshape((-1, ny, nx, 3, 9)), (wavelets_r.shape[0], 8, 20, 3, 9))
        w_i = skimage.transform.resize(wavelets_i.reshape((-1, ny, nx, 3, 9)), (wavelets_i.shape[0], 8, 20, 3, 9))
        phase = np.arctan(w_i / w_r)
        phase_diff = np.diff(phase, axis=0)
        cos = np.clip(w_r, a_min=0, a_max=None)
        cos_m = np.clip(-w_r, a_min=0, a_max=None)
        del w_r

        sin = np.clip(w_i, a_min=0, a_max=None)
        sin_m = np.clip(-w_i, a_min=0, a_max=None)
        del w_i

        sigma = 1
        # pn_wavelets = (np.cos(phi)*cos) + (np.cos(phi)* cos_m) + (np.sin(phi)*sin) + (np.sin(phi)*sin_m)
        pn_wavelets = cos + cos_m + sin + sin_m
        pn_wavelets_fwd = pn_wavelets * np.insert(np.clip(phase_diff, 0, None), [0], np.zeros((8, 20, 3, 9)),
                                                  0).reshape(9000, 8, 20, 3, 9)
        pn_wavelets_bkwd = pn_wavelets * np.insert(np.clip(-phase_diff, 0, None), [0], np.zeros((8, 20, 3, 9)),
                                                   0).reshape(9000, 8, 20, 3, 9)
        wavelets = np.stack([pn_wavelets, pn_wavelets_fwd, pn_wavelets_bkwd], axis=5)
        del pn_wavelets_bkwd, pn_wavelets_fwd, wavelets_r, wavelets_i
        gc.collect()

    wavelets = wavelets[:, :, :, scale, :, :]
    w = wavelets.reshape((9000, 8, 20, 9, 3))
    plt.figure()
    plt.plot(w[:, 5, 14, 4, 0])
    plt.plot(w[:, 5, 14, 4, 1])
    plt.plot(w[:, 5, 14, 4, 2])

    wavelets = wavelets.reshape(
        (wavelets.shape[0], wavelets.shape[1] * wavelets.shape[2] * wavelets.shape[3] * wavelets.shape[4]))
    return wavelets


def load_stimulus_simple_cell(path='/media/sophie/Expansion1/UCL/datatest/', downsampling=False):
    wavelets_r=np.load(path+'dwt_videodata_r.npy')
    wavelets_i = np.load(path+'dwt_videodata_i.npy')
    if downsampling:
        wavelets_r = skimage.transform.resize(abs(wavelets_r), (9000, 20, 8, 8), anti_aliasing=True)
        wavelets_r=np.swapaxes(wavelets_r, 2,1)

        wavelets_i = skimage.transform.resize(abs(wavelets_i), (9000, 20, 8, 8), anti_aliasing=True)
        wavelets_i=np.swapaxes(wavelets_i, 2,1)

    return wavelets_r,wavelets_i



def load_stimulus_simple_cell2(path='/media/sophie/Expansion1/UCL/datatest/', downsampling=False):
    wavelets_r=np.load(path+'dwt_videodata2_r.npy')[:9000]#, mmap_mode='c')
    wavelets_r=wavelets_r
    wavelets_i = np.load(path+'dwt_videodata2_i.npy')[:9000]#,  mmap_mode='c')
    if downsampling:
        wavelets_r = skimage.transform.resize(abs(wavelets_r), (9000, 20, 8, 8, 3), anti_aliasing=True)
        wavelets_r=np.swapaxes(wavelets_r, 2,1)

        wavelets_i = skimage.transform.resize(abs(wavelets_i), (9000, 20, 8, 8, 3), anti_aliasing=True)
        wavelets_i=np.swapaxes(wavelets_i, 2,1)

    return wavelets_r,wavelets_i


def loadExperiment(dirs, exp_info, pathdir,block_end, n_planes, n_repeat=6,n_frames=9000):
    exp_path = clu.find_expt_file(exp_info, 'root', dirs=dirs)
    exp_path=exp_info[0]+'/'+exp_info[1]
    tlfile = clu.find_expt_file(exp_info, 'timeline', dirs)
    tl = tlu.load_timeline(tlfile)

    # frame_times = tpu.get_frame_times(tl)
    input_ind = 'neuralFrames' == tlu.get_input_names(tl)
    tp = tl['rawDAQData'][:, input_ind].flatten()
    ind = np.diff(tp, prepend=tp[0]) > 0
    frame_times = tl['rawDAQTimestamps'][ind]
    frame_times=frame_times[:frame_times.shape[0]-(frame_times.shape[0]%n_planes)]#make it dividable by n_plane
    frame_times=frame_times.reshape((-1, n_planes))
    # frame_times=np.mean(frame_times, axis=1)

    neuron_pos = np.concatenate([np.asarray([sta['med'] for sta in np.load(
        dirs[0] + exp_path + '/suite2p/plane%d/stat.npy' % plane,
        allow_pickle=True)[np.load(
        dirs[0] + exp_path + '/suite2p/plane%d/iscell.npy' % plane)[:,
                           0].astype(bool)]]) for plane in range(n_planes)])

    n_cell = neuron_pos.shape[0]

    Nb_frames = n_frames * n_repeat
    verbose = False

    input_ind = 'photoDiode' == tlu.get_input_names(tl)
    syncEcho_thresh = 1.8
    esynv = tl['rawDAQData'][:, input_ind].flatten() > syncEcho_thresh
    syncEcho_flip = np.asarray(np.logical_or(
        np.logical_and(np.logical_not(esynv[:-1]), esynv[1:]),
        np.logical_and(np.logical_not(esynv[1:]), esynv[:-1])
    )).nonzero()[0]
    syncEcho_flip_times = tl['rawDAQTimestamps'][syncEcho_flip]
    print('syncEcho_flip_times: ', syncEcho_flip_times.shape)


    plt.figure()
    input_ind = 'photoDiode' == tlu.get_input_names(tl)
    plt.plot(tl['rawDAQData'][:, input_ind].flatten())
    # input_ind = 'syncEcho' == tlu.get_input_names(tl)
    # plt.plot(tl['rawDAQData'][:, input_ind].flatten())
    plt.twiny()
    plt.scatter(frame_times[:, 0], np.ones((frame_times.shape[0])), c='r')
    # plt.scatter(time_trial1[:, 1], np.ones((time_trial1.shape[0])), c='k')

    R=[]
    for plane in range(n_planes):
        print('plane :', plane)
        F = np.load(dirs[0]+exp_path+'/suite2p/plane%d/F.npy' % plane)[:, block_end:][np.load(dirs[0] + exp_path + '/suite2p/plane%d/iscell.npy' % plane)[:,0].astype(bool)]
        Fneu = np.load(dirs[0] + exp_path + '/suite2p/plane%d/Fneu.npy' % plane)[:, block_end:][
                                   np.load(
                                       dirs[0]+exp_path+'/suite2p/plane%d/iscell.npy' % plane)[
                                   :, 0].astype(bool)]

        spks = F - (0.7 * Fneu)




        # f=frame_times[:, plane]
        # starttrial = f[f >= syncEcho_flip_times[0]]
        #
        # trial1 = np.logical_and(f >= syncEcho_flip_times[0], f < syncEcho_flip_times[Nb_frames])
        # time_trial1 = f[trial1]
        #
        # trial2 = np.logical_and(f >= syncEcho_flip_times[Nb_frames],
        #                         f < syncEcho_flip_times[Nb_frames * 2])
        # time_trial2 = f[trial2]
        #
        # trial3 = np.logical_and(f >= syncEcho_flip_times[Nb_frames * 2],
        #                         f < syncEcho_flip_times[Nb_frames * 3])
        # time_trial3 = f[trial3]
        #
        # trial4 = np.logical_and(f >= syncEcho_flip_times[Nb_frames * 3],
        #                         f < syncEcho_flip_times[Nb_frames * 4])
        # time_trial4 = f[trial4]
        #
        # trial5 = np.logical_and(f >= syncEcho_flip_times[Nb_frames * 4],
        #                         f < syncEcho_flip_times[Nb_frames * 5])
        # time_trial5 = f[trial5]


        # trials = [trial1, trial2, trial3, trial4, trial5]
        # time_trials = [time_trial1, time_trial2, time_trial3, time_trial4, time_trial5]
        window = [1.15]
        # avg_spks=np.mean(np.array([spks[:,np.asarray(trial!=0).nonzero()[0]].tolist()[0] for trial in trials]), axis=0)
        resps_all = []
        # for i, trial in enumerate(trials):
        spks_rt_noz = spks[:, :frame_times.shape[0]]
        spks_rt = utils.zscore(spks_rt_noz, ax=1, epsilon=1e-5)
        spks_rt = np.array([spks_rt[:, i] - np.min(spks_rt, axis=1) for i in range(spks_rt.shape[1])]).T
        # print(i)
        # print(np.sum(trial), len(time_trials[i]), spks_rt.shape)
        # resps_all.append([utils.interp_event_responses(frame_times, spks_rt,
        #                                                events=syncEcho_flip_times[Nb_frames * i:Nb_frames * (i + 1)],
        #                                                window=window, mean_over_window=False, print_interval=None)])
        try:
            resps_all = utils.interp_event_responses(frame_times[:, plane], spks_rt, events=syncEcho_flip_times,
                                                 window=window,
                                                 mean_over_window=False, print_interval=None)
        except:
            resps_all = utils.interp_event_responses(frame_times[:spks_rt.shape[1], plane], spks_rt, events=syncEcho_flip_times,
                                                     window=window,
                                                     mean_over_window=False, print_interval=None)
        print(np.array(resps_all).shape)
        if plane==0:
            R= np.array(resps_all)
        else:
            R=np.concatenate((R, np.array(resps_all)), axis=1)

    plt.figure()
    plt.scatter(syncEcho_flip_times, np.ones(syncEcho_flip_times.shape[0]))
    plt.scatter(syncEcho_flip_times, resps_all[:, 0], c='g')

    resps_all = np.nan_to_num(R)
    # resps_all = resps_all[:, 0, :, :, 0]

    return resps_all, neuron_pos, syncEcho_flip_times


def align_datas(exp_info, dirs,spks, Nb_frames, nb_plane=1, plane=-1, w=0.0, threshold=1.25, methods='frame2ttl'):
    tlfile = clu.find_expt_file(exp_info, 'timeline', dirs)
    tl = tlu.load_timeline(tlfile)

    # frame_times = tpu.get_frame_times(tl)
    try:
        input_ind = 'neuralFrames' == tlu.get_input_names(tl)
        # print(input_ind)
        tp = tl['rawDAQData'][:, input_ind].flatten()
        ind = np.diff(tp, prepend=tp[0]) > 0
        frame_times = tl['rawDAQTimestamps'][ind]

        input_ind = 'photoDiode' == tlu.get_input_names(tl)
        syncEcho_thresh = 1.5
    except:
        if methods=='photosensor':
            syncEcho_thresh = threshold#1.25

        elif methods=='frame2ttl':
            syncEcho_thresh = threshold
        else:
            print('unknown timeline variable')

        print(methods, syncEcho_thresh)
        input_ind = 'neural_frames' == tlu.get_input_names(tl)
        # print(input_ind)
        tp = tl['rawDAQData'][:, input_ind].flatten()
        ind = np.diff(tp, prepend=tp[0]) > 0
        frame_times = tl['rawDAQTimestamps'][ind]

        input_ind = methods == tlu.get_input_names(tl)

    esynv = tl['rawDAQData'][:, input_ind].flatten() > syncEcho_thresh
    syncEcho_flip = np.asarray(np.logical_or(
        np.logical_and(np.logical_not(esynv[:-1]), esynv[1:]),
        np.logical_and(np.logical_not(esynv[1:]), esynv[:-1])
    )).nonzero()[0]
    syncEcho_flip_times = tl['rawDAQTimestamps'][syncEcho_flip]
    print('syncEcho_flip_times: ', syncEcho_flip_times.shape)

    if nb_plane!=1:
        print('multiple planes')
        frame_times=frame_times[(frame_times.shape[0]%nb_plane):].reshape(-1, nb_plane)
        print(frame_times.shape)
        if plane==-1:
            frame_times=np.mean(frame_times, axis=1)
        else:
            frame_times = frame_times[:, plane]
        print(frame_times.shape)

    starttrial = frame_times[frame_times >= syncEcho_flip_times[0]]
    # Nb_frames = 9000
    trials=[]
    time_trials=[]
    tt=True
    t=1
    while tt:
        try:
            if t==1:
                print('trial', t)
                trial1 = np.logical_and(frame_times >= syncEcho_flip_times[0], frame_times < syncEcho_flip_times[Nb_frames*t])
                time_trial1 = frame_times[trial1]
                t=t+1
                print(trial1.shape)
            else:
                print('trial', t)
                trial1 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames*(t-1)],
                                        frame_times < syncEcho_flip_times[Nb_frames * t])
                time_trial1 = frame_times[trial1]
                t = t + 1
            trials.append(trial1)
            time_trials.append(time_trial1)

        except:
            print('incomplete trial')
            tt=False
            trial1=np.zeros(trials[0].shape)
            temp=np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames*(t-1)],
                                        frame_times < syncEcho_flip_times[np.minimum(Nb_frames * t, syncEcho_flip_times.shape[0]-1)])
            trial1[:temp.shape[0]] = temp
            trial1=trial1.astype(bool)
            time_trial1 = frame_times[trial1]

            if time_trial1.shape!=(0,):
                trials.append(trial1)
                time_trials.append(time_trial1)



    # print('trial1')
    # trial1 = np.logical_and(frame_times >= syncEcho_flip_times[0], frame_times < syncEcho_flip_times[Nb_frames])
    # time_trial1 = frame_times[trial1]
    #
    # print('trial2')
    # trial2 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames],
    #                         frame_times < syncEcho_flip_times[Nb_frames * 2])
    # time_trial2 = frame_times[trial2]
    # print('trial3')
    # trial3 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * 2],
    #                         frame_times < syncEcho_flip_times[Nb_frames * 3])
    # time_trial3 = frame_times[trial3]
    # print('trial4')
    # trial4 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * 3],
    #                         frame_times < syncEcho_flip_times[Nb_frames * 4])
    # time_trial4 = frame_times[trial4]
    #
    # trial5 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * 4],
    #                         frame_times < syncEcho_flip_times[Nb_frames * 5])
    # time_trial5 = frame_times[trial5]

    # trial6 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * 5],
    #                         frame_times < syncEcho_flip_times[Nb_frames * 7])
    # time_trial6 = frame_times[trial6]

    # trials = [trial1, trial2, trial3, trial4, trial5]
    # time_trials = [time_trial1, time_trial2, time_trial3, time_trial4, time_trial5]
    window = [w]
    # avg_spks=np.mean(np.array([spks[:,np.asarray(trial!=0).nonzero()[0]].tolist()[0] for trial in trials]), axis=0)
    resps_all = []
    for i, trial in enumerate(trials):
        print(i)
        spks_rt = utils.zscore(spks[:, np.asarray(trial != 0).nonzero()[0]], ax=1, epsilon=1e-5)
        spks_rt = np.array([spks_rt[:, i] - np.min(spks_rt, axis=1) for i in range(spks_rt.shape[1])]).T
        print(np.sum(trial), len(time_trials[i]), spks_rt.shape)
        temp=np.zeros((Nb_frames, spks.shape[0], 1))
        temp1=utils.interp_event_responses(time_trials[i], spks_rt,
                                                       events=syncEcho_flip_times[Nb_frames * i:Nb_frames * (i + 1)],
                                                       window=window, mean_over_window=False, print_interval=None)
        temp[:temp1.shape[0]]=temp1
        resps_all.append([temp])

    return resps_all

def loadSPKMesoscope(exp_info, dirs, path, block_end, Nb_plane=3, Nb_frames=9000, first=False, threshold=1.25,plane=-1,  method='frame2ttl'):
    if first:
        if plane != -1:
            spks = np.load(
                path + '/plane%d/spks.npy' % plane)[
                                       np.load(
                                           path + '/plane%d/iscell.npy' % plane)[
                                       :, 0].astype(bool)][:, :block_end]
        else:
            spks = np.concatenate([np.load(
                path + '/plane%d/spks.npy' % p)[
                                       np.load(
                                           path + '/plane%d/iscell.npy' % p)[
                                       :, 0].astype(bool)] for p in range(Nb_plane)])[:, :block_end]

    else:
        if plane != -1:
            spks = np.load(
                path + '/plane%d/spks.npy' % plane)[np.load(path + '/plane%d/iscell.npy' % plane)[
                                       :, 0].astype(bool)][:, block_end:]
        else:
            spks = np.concatenate([np.load(
                path + '/plane%d/spks.npy' % p)[
                                       np.load(
                                           path + '/plane%d/iscell.npy' % p)[
                                       :, 0].astype(bool)] for p in range(Nb_plane)])[:, block_end:]

    if plane != -1:
        neuron_pos = np.array([(1, plane * 512) + np.asarray([sta['med'] for sta in np.load(
            path + '/plane%d/stat.npy' % plane,
            allow_pickle=True)[np.load(
            path + '/plane%d/iscell.npy' % plane)[:,
                               0].astype(bool)]])])[0]
    else:
        neuron_pos = np.concatenate([(1, p * 512) + np.asarray([sta['med'] for sta in np.load(
            path + '/plane%d/stat.npy' % p,
            allow_pickle=True)[np.load(
            path + '/plane%d/iscell.npy' % p)[:,
                               0].astype(bool)]]) for p in range(Nb_plane)])

    print('shape spks : ', spks.shape)
    print('neuron_pos spks : ', neuron_pos.shape)


    resps_all=align_datas(exp_info, dirs, spks, Nb_frames,nb_plane=Nb_plane, threshold=threshold, plane=plane, methods=method)
    resps_all = np.array(resps_all)
    resps_all = np.nan_to_num(resps_all)
    resps_all = resps_all[:, 0, :, :, 0]
    return resps_all, neuron_pos




def loadRespMesoscope(exp_info, dirs, path, block_end):
    tlfile = clu.find_expt_file(exp_info, 'timeline', dirs)
    tl = tlu.load_timeline(tlfile)

    # frame_times = tpu.get_frame_times(tl)
    input_ind = 'neuralFrames' == tlu.get_input_names(tl)
    tp = tl['rawDAQData'][:, input_ind].flatten()
    ind = np.diff(tp, prepend=tp[0]) > 0
    frame_times = tl['rawDAQTimestamps'][ind]

    spks_deconvolved = np.concatenate([np.load(
       path + '/plane%d/spks.npy' % plane)[
                                           np.load(
                                               path+'/plane%d/iscell.npy' % plane)[
                                           :, 0].astype(bool)] for plane in range(3)])[:, block_end:]

    F = np.concatenate([np.load(
        path+'/plane%d/F.npy' % plane)[np.load(
        path+'/plane%d/iscell.npy' % plane)[:,
                                                                                                               0].astype(
        bool)] for plane in range(3)])[:, block_end:]
    Fneu = np.concatenate([np.load(
        path+'/plane%d/Fneu.npy' % plane)[
                               np.load(
                                   path+'/plane%d/iscell.npy' % plane)[
                               :, 0].astype(bool)] for plane in range(3)])[:, block_end:]

    spks = F - (0.7 * Fneu)

    neuron_pos = np.concatenate([(1, plane * 512) + np.asarray([sta['med'] for sta in np.load(
        path+'/plane%d/stat.npy' % plane,
        allow_pickle=True)[np.load(
        path+'/plane%d/iscell.npy' % plane)[:,
                           0].astype(bool)]]) for plane in range(3)])

    n_cell = neuron_pos.shape[0]

    Nb_frames = 9000
    verbose = False

    input_ind = 'photoDiode' == tlu.get_input_names(tl)
    syncEcho_thresh = 1.5
    esynv = tl['rawDAQData'][:, input_ind].flatten() > syncEcho_thresh
    syncEcho_flip = np.asarray(np.logical_or(
        np.logical_and(np.logical_not(esynv[:-1]), esynv[1:]),
        np.logical_and(np.logical_not(esynv[1:]), esynv[:-1])
    )).nonzero()[0]
    syncEcho_flip_times = tl['rawDAQTimestamps'][syncEcho_flip]
    print('syncEcho_flip_times: ', syncEcho_flip_times.shape)

    starttrial = frame_times[frame_times >= syncEcho_flip_times[0]]

    trial1 = np.logical_and(frame_times >= syncEcho_flip_times[0], frame_times < syncEcho_flip_times[Nb_frames])
    time_trial1 = frame_times[trial1]

    trial2 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames],
                            frame_times < syncEcho_flip_times[Nb_frames * 2])
    time_trial2 = frame_times[trial2]

    trial3 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * 2],
                            frame_times < syncEcho_flip_times[Nb_frames * 3])
    time_trial3 = frame_times[trial3]

    trial4 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * 3],
                            frame_times < syncEcho_flip_times[Nb_frames * 4])
    time_trial4 = frame_times[trial4]

    trial5 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * 4],
                            frame_times < syncEcho_flip_times[Nb_frames * 5])
    time_trial5 = frame_times[trial5]

    trial6 = np.logical_and(frame_times >= syncEcho_flip_times[Nb_frames * 5],
                            frame_times < syncEcho_flip_times[Nb_frames * 7])
    time_trial6 = frame_times[trial6]

    trials = [trial1, trial2, trial3, trial4, trial5]
    time_trials = [time_trial1, time_trial2, time_trial3, time_trial4, time_trial5]
    window = [0.15]
    # avg_spks=np.mean(np.array([spks[:,np.asarray(trial!=0).nonzero()[0]].tolist()[0] for trial in trials]), axis=0)
    resps_all = []
    for i, trial in enumerate(trials):
        spks_rt = utils.zscore(spks[:, np.asarray(trial != 0).nonzero()[0]], ax=1, epsilon=1e-5)
        spks_rt = np.array([spks_rt[:, i] - np.min(spks_rt, axis=1) for i in range(spks_rt.shape[1])]).T
        print(i)
        print(np.sum(trial), len(time_trials[i]), spks_rt.shape)
        resps_all.append([utils.interp_event_responses(time_trials[i], spks_rt,
                                                       events=syncEcho_flip_times[Nb_frames * i:Nb_frames * (i + 1)],
                                                       window=window, mean_over_window=False, print_interval=None)])
    resps_all = np.array(resps_all)
    resps_all = np.nan_to_num(resps_all)
    resps_all = resps_all[:, 0, :, :, 0]

    U, S, Vh = np.linalg.svd(resps_all.reshape(-1, n_cell), full_matrices=False)
    A = U[:, 2:] @ np.diag(S[2:]) @ Vh[2:, :]

    resps_all = A.reshape((5, 9000, n_cell))

    return resps_all, neuron_pos, time_trial1




def get_rfs(i, r2, r1, i2, i1,y2, y1, neuron_pos):
    RFS = []
    print('r2')
    for o in range(8):
        # print(o)
        # convolved_wavelets=convolved_wavelets_r.reshape(9000, 54, 135, 8, 4)[4500:, :, :, o, :]
        rfs = PearsonCorrelationPinkNoise(r2[:, :, :, o].reshape(4500, -1), y2[0].reshape(-1, 1), neuron_pos[i],
                                          135, 54)
        rfs = rfs[0]
        RFS.append(rfs)
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    RFS = np.array(RFS)
    RFS = np.moveaxis(RFS, 0, 3)
    RFS_r_2 = RFS[0]
    del RFS, rfs
    gc.collect()

    RFS = []
    print('r1')
    for o in range(8):
        # print(o)
        # convolved_wavelets = convolved_wavelets_r.reshape(9000, 54, 135, 8, 4)[:4500, :, :, o, :]
        # rfs = PearsonCorrelationPinkNoise(convolved_wavelets.reshape(4500, -1),
        #                                   resps_all_mean[:4500, idx].reshape(-1, 1), neuron_pos[idx], 135, 54)
        rfs = PearsonCorrelationPinkNoise(r1[:, :, :, o].reshape(4500, -1), y1[0].reshape(-1, 1),
                                          neuron_pos[i], 135, 54)
        rfs = rfs[0]
        RFS.append(rfs)
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    RFS = np.array(RFS)
    RFS = np.moveaxis(RFS, 0, 3)
    RFS_r_1 = RFS[0]
    del RFS, rfs
    gc.collect()

    RFS = []
    print('i2')
    for o in range(8):
        # print(o)
        # convolved_wavelets = convolved_wavelets_i.reshape(9000, 54, 135, 8, 4)[4500:, :, :, o, :]
        # rfs = PearsonCorrelationPinkNoise(convolved_wavelets.reshape(4500, -1),
        #                                   resps_all_mean[4500:, idx].reshape(-1, 1), neuron_pos[idx], 135, 54)
        rfs = PearsonCorrelationPinkNoise(i2[:, :, :, o].reshape(4500, -1), y2[0].reshape(-1, 1),
                                          neuron_pos[i], 135, 54)
        rfs = rfs[0]
        RFS.append(rfs)
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    RFS = np.array(RFS)
    RFS = np.moveaxis(RFS, 0, 3)
    RFS_i_2 = RFS[0]
    del RFS, rfs
    gc.collect()

    RFS = []
    print('i1')
    for o in range(8):
        # print(o)
        # convolved_wavelets = convolved_wavelets_i.reshape(9000, 54, 135, 8, 4)[:4500, :, :, o, :]
        # rfs = PearsonCorrelationPinkNoise(convolved_wavelets.reshape(4500, -1),
        #                                   resps_all_mean[:4500, idx].reshape(-1, 1), neuron_pos[idx], 135, 54)
        rfs = PearsonCorrelationPinkNoise(i1[:, :, :, o].reshape(4500, -1), y1[0].reshape(-1, 1),
                                          neuron_pos[i], 135, 54)
        rfs = rfs[0]
        RFS.append(rfs)
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    RFS = np.array(RFS)
    RFS = np.moveaxis(RFS, 0, 3)
    RFS_i_1 = RFS[0]
    del RFS, rfs
    gc.collect()
    # RFS = RFS_i_1
    # fig, ax = plt.subplots(8, 4)
    # for i in range(8):
    #     vmax=np.max(RFS)
    #     vmin = np.min(RFS)
    #     ax[i, 0].imshow(RFS[:, :, i, 0].T,vmin=vmin, vmax=vmax, cmap='coolwarm')
    #     ax[i, 1].imshow(RFS[:, :, i, 1].T,vmin=vmin, vmax=vmax, cmap='coolwarm')
    #     ax[i, 2].imshow(RFS[:, :, i, 2].T, vmin=vmin, vmax=vmax,cmap='coolwarm')
    #     ax[i, 3].imshow(RFS[:, :, i, 3].T, vmin=vmin, vmax=vmax,cmap='coolwarm')

    RFS = np.array([RFS_r_1, RFS_r_2, RFS_i_1, RFS_i_2])
    np.save('/media/sophie/Expansion1/UCL/datatest/SP045/2023-10-04/3/rfs/rfs_%d_HR.npy' % i, RFS)


def plotRFS(RFSs):
    vmax = np.max(RFSs)
    vmin = -np.max(RFSs)
    print(vmin, vmax)
    for RFS in RFSs:
        fig, ax = plt.subplots(8, 4)
        for i in range(8):

            ax[i, 0].imshow(RFS[:, :, i, 0].T,vmin=vmin, vmax=vmax, cmap='coolwarm')
            ax[i, 1].imshow(RFS[:, :, i, 1].T,vmin=vmin, vmax=vmax, cmap='coolwarm')
            ax[i, 2].imshow(RFS[:, :, i, 2].T, vmin=vmin, vmax=vmax,cmap='coolwarm')
            ax[i, 3].imshow(RFS[:, :, i, 3].T, vmin=vmin, vmax=vmax,cmap='coolwarm')



    (a, x, y, o, s)=np.where(RFSs==np.max(RFSs))
    print(a)
    print('xmax =',  x)
    print('ymax =', y)
    print('orientation =', o*180/8)
    print('scale =', s)
    print('phase =', np.arctan(RFSs[1, x, y, o, s]/RFSs[3, x, y, o, s])*180/np.pi)
    return (a, x, y, o, s)



def splitDataset(x_i, x_r, y):
    BATCH_SIZE = 1
    Dataset = TensorDataset(torch.Tensor(y[4500:]).T, torch.Tensor(y[:4500]).T)
    r2 = torch.Tensor(x_r.reshape(9000, 135, 54, 8, 4)[4500:])
    r1 = torch.Tensor(x_r.reshape(9000, 135, 54, 8, 4)[:4500])
    i2 = torch.Tensor(x_i.reshape(9000, 135, 54, 8, 4)[4500:])
    i1 = torch.Tensor(x_i.reshape(9000, 135, 54, 8, 4)[:4500])

    DL = DataLoader(Dataset, shuffle=False, batch_size=BATCH_SIZE)
    return r2, r1, i2, i1, DL



def getRFS_idx(idx, x_i, x_r, y, neuron_pos):
    r2, r1, i2, i1, DL = splitDataset(x_i, x_r, y)
    get_rfs(idx, r2, r1, i2, i1, torch.Tensor(y[4500:]).T, torch.Tensor(y[:4500]).T, neuron_pos)


def GetRFS_allneurons(x_i, x_r, y, startid , neuron_pos):
    # R2=TensorDataset(torch.Tensor(convolved_wavelets_r.reshape(9000, 54, 135, 8, 4)[4500:]), torch.Tensor(resps_all_mean[4500:]) )
    # R1=TensorDataset(torch.Tensor(convolved_wavelets_r.reshape(9000, 54, 135, 8, 4)[:4500]), torch.Tensor(resps_all_mean[:4500]) )
    # I2=TensorDataset(torch.Tensor(convolved_wavelets_i.reshape(9000, 54, 135, 8, 4)[4500:]), torch.Tensor(resps_all_mean[4500:]) )
    # I1=TensorDataset(torch.Tensor(convolved_wavelets_i.reshape(9000, 54, 135, 8, 4)[:4500]),torch.Tensor(resps_all_mean[:4500]) )
    # R1_DL = DataLoader(R1, shuffle=False, batch_size=BATCH_SIZE)
    # R2_DL = DataLoader(R2, shuffle=False, batch_size=BATCH_SIZE)
    # I1_DL = DataLoader(I1, shuffle=False, batch_size=BATCH_SIZE)
    # I2_DL = DataLoader(I2, shuffle=False, batch_size=BATCH_SIZE)

    r2, r1, i2, i1, DL=splitDataset(x_i, x_r, y)


    for i, (y2, y1) in enumerate(DL):

        if i >= startid:
            print(i)
            get_rfs(i, r2, r1, i2, i1, y2, y1, neuron_pos)

