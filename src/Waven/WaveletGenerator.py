"""
Created on Wed Mar 25 19:31:32 2025

@author: Sophie Skriabine
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as sio
import skimage
from skimage import transform
import os
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')  # Use Agg backend for headless/non-GUI environments
from skimage.measure import block_reduce
import cv2
import gc
import torch
from tqdm import tqdm

from skimage.filters import gabor_kernel




def makeGaborFilter(i, j, angle, sigma, phase, f=0.4, lx=54, ly=135, plot=False, freq=True):
    backgrd=np.zeros((lx, ly))
    if freq:
        gk = gabor_kernel(frequency=f, theta=angle, sigma_x=sigma, sigma_y=sigma, offset=phase)
    else:
        gk = gabor_kernel(frequency=(-0.016*sigma)+0.148, theta=angle, sigma_x=sigma, sigma_y=sigma,offset=phase)
    # plt.figure()
    # plt.imshow(gk.real)
    #
    # plt.figure()
    # plt.imshow(canvas, vmin=0, vmax=0.006)

    canvas=np.ones((lx+(2*gk.shape[0]), ly+(2*gk.shape[1])))
    canvas[gk.shape[0]:gk.shape[0]+lx, gk.shape[1]:gk.shape[1]+ly]=backgrd

    dp=(gk.shape[0]-1)/2

    x=i+gk.shape[0]
    y=j+gk.shape[1]

    canvas[int(x-dp):int(x+dp+1), int(y-dp):int(y+dp+1)]=gk.real
    backgrd=canvas[gk.shape[0]:gk.shape[0]+lx, gk.shape[1]:gk.shape[1]+ly]
    if plot:
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.imshow(backgrd.T, cmap='Greys')
    return backgrd.T.astype('float16')



def makeGaborFilter3D(i, j, angle, sigma, tp_w, f=0.4, lx=54, ly=135, alpha1=0, alpha2=np.pi/4):

    phases=np.linspace(alpha1, alpha2, tp_w)
    # print(phases)
    f3d=np.array([ makeGaborFilter(i, j, angle, sigma, phase, f=f, lx=lx, ly=ly) for phase in phases])
    return f3d.astype('float16')


def makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, frequencies):
    library=[]
    lx=xs.shape[0]
    ly=ys.shape[0]
    for x in xs:
        print(x)
        for y in ys:
            for t in thetas:
                for s in sigmas:
                    for f in frequencies:
                        for o in offsets:
                            library.append( makeGaborFilter(x, y, t, s, o, f, lx=lx, ly=ly, freq=True))

    library=np.array(library)
    return library.reshape((lx, ly, thetas.shape[0], sigmas.shape[0], frequencies.shape[0], offsets.shape[0], -1))

def makeFilterLibrary(xs, ys, thetas, sigmas, offsets, f, freq=True):
    """
    builds the Gabor library

    Parameters:
        thetas (int): number of orientatuion equally spaced between 0 and 180 degree.
    	Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
    	f (list): spatial frequencies expressed in pixels per cycles.
    	offsets (list): 0 and pi/2.
    	xs (int): number of azimuth positions (pix) (x shape of the downsampled stimuli).
    	ys (int): number of elevation positions (pix) (y shape of the downsampled stimuli).
    	freq (boolean): if True the, takes into account the frequencies list to generate the gabors filters, if False, there is a linear relationship between the size and the spatial frequencies as found in ref paper

    Returns:
        npy file containing all the generated gabor filters of shape (nx, ny, n_orientation, n_sizes, n_freq (if defined independantly from sizes, n_phases, nx*ny))
    """
    library=[]
    lx=xs.shape[0]
    ly=ys.shape[0]
    for x in tqdm(xs):
        for y in ys:
            for t in thetas:
                for s in sigmas:
                    for o in offsets:
                        library.append( makeGaborFilter(x, y, t, s, o, f, lx=lx, ly=ly, freq=freq))

    library=np.array(library)
    return library.reshape((lx, ly, thetas.shape[0], sigmas.shape[0], offsets.shape[0], -1))



import itertools
def makeFilterLibrary3D(xs, ys, thetas, sigmas, offsets, f, tp_w,  alpha1, alpha2, filename):
    # library=[]
    lx = xs.shape[0]
    ly = ys.shape[0]
    fp = np.zeros( shape=(lx, ly, thetas.shape[0], sigmas.shape[0], tp_w,ly, lx), dtype='float16')
    print(fp.shape)
    i=0
    # with open(filename, mode="wb") as fp:
    for x in xs:
        print(x)
        for y in ys:
            for i, t in enumerate(thetas):
                for j, s in enumerate(sigmas):
                        l = makeGaborFilter3D(x, y, t, s, tp_w, f, lx=lx, ly=ly,  alpha1=alpha1, alpha2=alpha2)
                        # print(l.shape)
                        fp[x, y, i, j]=l

    print('saving...')
    np.save(filename,  fp)
    return fp



def waveletTransform(frame,phase, L):
    output=L[:, :, :,phase]@torch.Tensor(frame.flatten()).cuda()
    # output=torch.sum(output, axis=(0, 1))
    return output.detach().cpu().numpy()


def waveletTransform3D(frame, L):
    output=L@torch.Tensor(frame.flatten()).cuda()
    # output=torch.sum(output, axis=(0, 1))
    return output.detach().cpu().numpy()


def getTrueRF(idx, rfs, L):
    rf=rfs[idx, :, :, :]#.swapaxes(0, 1)
    # rf = skimage.transform.resize(rf, (135, 54, 8),order=5, anti_aliasing=True)
    rfv=rf.reshape(1, -1)@L[:, :, :, 2, 0, :].reshape(-1,7290)

    plt.figure()
    plt.imshow(rfv.reshape(54, 135)[5:-5, 5:-5],  vmin=-np.max(rfv), vmax=np.max(rfv) ,cmap='coolwarm')#vmin=-0.0014, vmax=0.0014,



def getWTfromNPY(videodata, waveletLibrary, phase, device='cuda'):
    WT = []
    l = torch.Tensor(waveletLibrary).to(device)
    for i, frame in tqdm(enumerate(videodata), total=len(videodata), desc=f"Wavelet transform (phase={phase})"):
        wt = waveletTransform(frame, phase, l)
        torch.cuda.empty_cache()
        WT.append(wt)
    WT = np.array(WT)
    # l = l.detach().cpu().numpy()
    # torch.cuda.empty_cache()
    # del l
    # gc.collect()
    return WT




def getWTfromNPY3D(videodata, waveletLibrary, tp_w, device='cuda'):
    WT = []
    l = torch.Tensor(waveletLibrary).to(device)
    for i in range(tp_w, videodata.shape[0]):
        print(i)
        wt = waveletTransform3D(videodata[i-tp_w:i], l)
        torch.cuda.empty_cache()
        WT.append(wt)
    WT = np.array(WT)
    # l = l.detach().cpu().numpy()
    # torch.cuda.empty_cache()
    # del l
    # gc.collect()
    return WT






def downsample_video_binary(path, visual_coverage, analysis_coverage, shape=(54, 135), chunk_size=1000,ratios=(1, 1)):
    """
    Downsample the video stimulus.

    Parameters:
        Path: path to the stimulus (.mp4)
        Visual Coverage (list): [azimuth left, azimuth right, elevation top , elevation bottom] in visual degree.
    	Analysis Coverage (list): [azimuth left, azimuth right, elevation top , elevation bottom] in visual degree.
        Shape (nx, ny): downsampled size
        chunk size: for precessing effisciency, default 1000
        ratio: if part of the screen is ignored

    Returns:
        saves the downsampled file at path
    """
    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    ret1=True
    F=[]
    r=0
    f = 0
    ratio_x, ratio_y=ratios
    print(f"Processing video with ratios: {ratio_x}, {ratio_y}")
    
    # Convert to numpy arrays for element-wise operations
    visual_coverage = np.array(visual_coverage)
    analysis_coverage = np.array(analysis_coverage)
    
    while ret:
        frames = []
        print(f"Processing chunk {r}") 
        ret1=ret
        i = 0
        while ret1:
            ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                if f<(r+1)*chunk_size:
                    if f >= r * chunk_size:
                        frames.append(img)
                        i=i+1
                        f = f + 1
                if f>=(r+1)*chunk_size:
                    ret1=False
                    print(f'Chunk full: frame {f}, count {i}')
                    print(f'Frames collected: {len(frames)}')
            else:
                print(f'Video ended at frame {f}, count {i}, ret={ret}')
                ret1=ret
        
        if len(frames) == 0:
            print('No more frames to process')
            break
            
        try:
            video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
            print(f'Video chunk shape: {video.shape}')
            video = video[:, :, :, 0]
            video_bin = video > 100
            del frames, video
            gc.collect()
            
            print(f'Binary video shape: {video_bin.shape}')
            
            # Calculate crop indices correctly
            xi = int((visual_coverage[2] - analysis_coverage[2]) * video_bin.shape[1] / (visual_coverage[2] - visual_coverage[3]))
            xe = int(xi + ratio_y * video_bin.shape[1])
            yi = int((visual_coverage[0] - analysis_coverage[0]) * video_bin.shape[2] / (visual_coverage[0] - visual_coverage[1]))
            ye = int(yi + ratio_x * video_bin.shape[2])
            
            print(f'Crop indices - xi:{xi}, xe:{xe}, yi:{yi}, ye:{ye}')
            
            video_bin = video_bin[:, xi:xe, yi:ye]
            video_bin = skimage.transform.resize(video_bin, (video_bin.shape[0], shape[0], shape[1]))
            video_binary = video_bin >= 0.5
            F.append(video_binary)
            del video_bin
            gc.collect()
            r = r + 1
        except Exception as e:
            print(f'Error processing chunk: {e}')
            break
    
    if len(F) > 0:
        video_downsampled = np.concatenate(F, axis=0)
        print(f'Final downsampled video shape: {video_downsampled.shape}')
        np.save(path[:-4]+'_downsampled.npy', video_downsampled.astype('bool'))
        print(f'Saved downsampled video to {path[:-4]}_downsampled.npy')
    else:
        print('ERROR: No frames were processed successfully')
        raise ValueError('Video processing failed - no frames were downsampled')


def downsample_video_uint(path, shape=(54, 135), chunk_size=1000):
    ## chunk size should be a divisor of the video total nb of frames
    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
    video = video[:, :, :, 0]
    nb_chunks=int(video.shape[0]/chunk_size)
    # video_bin=video
    del frames
    gc.collect()
    # video_bin=video
    frames = []
    for i in range(nb_chunks):
        print(i)
        video_bin = skimage.transform.resize(video[i * chunk_size:(i + 1) * chunk_size], (chunk_size, shape[0], shape[1]))  # 137
        frames.append(video_bin)
        del video_bin
        gc.collect()
    video_downsampled = np.concatenate(frames, axis=0)
    np.save(path[:-4]+'_downsampled.npy', video_downsampled)

def waveletDecomposition(videodata, phase, sigmas, folder_path, library_path='/media/sophie/Expansion1/UCL/datatest/gabors_library.npy', device='cuda:0'):
    """
    Runs the wavelet decomposition

    Parameters:
        videodata (array like): downsampled stimulus movie (npy).
        Phases (list): 0 and pi/2.
    	Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
    	folder_path: Path where to save the decomposition
        Library Path: path to Gabor library (same as save path if ran)

    Returns:
        saves the wavelet decomposition as 'dwt_videodata_0 / 1.npy' at folder_path
    """
    L = np.load(library_path)
    WT = []
    for s, ss in enumerate(sigmas):
        l = L[:, :, :, s]
        wt = getWTfromNPY(videodata, l, phase, device=device)
        WT.append(wt)
    WT = np.array(WT)
    WT = np.moveaxis(WT, 0, 4)
    np.save(folder_path+'/dwt_videodata_'+str(phase)+'.npy', WT)

