import Waven.WaveletGenerator as wg
import Waven.Analysis_Utils as au
import Waven.LoadPinkNoise as lpn
import numpy as np
import gc
import os
import torch
import matplotlib.pyplot as plt
import argparse
import json

"""
Run analysis without GUI - suitable for Jupyter notebooks and headless environments
"""

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Zebra Analysis Pipeline')
    parser.add_argument('--param_defaults', type=str, required=True,
                       help='JSON string or file path containing param_defaults dictionary')
    parser.add_argument('--gabor_param', type=str, required=True,
                       help='JSON string or file path containing gabor_param dictionary')
    
    args = parser.parse_args()
    
    # Load parameters from JSON strings or files
    try:
        # Try to parse as JSON string first
        param_defaults = json.loads(args.param_defaults)
    except json.JSONDecodeError:
        # If that fails, try to load as file path
        with open(args.param_defaults, 'r') as f:
            param_defaults = json.load(f)
    
    try:
        gabor_param = json.loads(args.gabor_param)
    except json.JSONDecodeError:
        with open(args.gabor_param, 'r') as f:
            gabor_param = json.load(f)
    
    # Extract parameters
    dirs = [param_defaults["Dirs"]]
    sigmas = np.array(eval(param_defaults["Sigmas"]))
    visual_coverage = eval(param_defaults["Visual Coverage"])
    analysis_coverage = eval(param_defaults["Analysis Coverage"])
    n_planes = int(param_defaults["Number of Planes"])
    block_end = int(param_defaults["Block End"])
    screen_x = int(param_defaults["screen_x"])
    screen_y = int(param_defaults["screen_y"])
    nx0 = int(param_defaults["NX0"])
    ny0 = int(param_defaults["NY0"])
    nx = int(param_defaults["NX"])
    ny = int(param_defaults["NY"])
    ns = len(sigmas)
    resolution = float(param_defaults["Resolution"])
    spks_path = param_defaults["Spks Path"]
    nb_frames = int(param_defaults["Number of Frames"])
    n_trial2keep = int(param_defaults["Number of Trials to Keep"])
    movpath = param_defaults["Movie Path"]
    lib_path = param_defaults["Library Path"]
    n_theta = int(gabor_param["N_thetas"])
    #set device
    try:
        device = param_defaults["Device"]
    except KeyError:
        device = "cuda:0"  # default to first CUDA device if not specified

    screen_ratio = abs(visual_coverage[0] - visual_coverage[1]) / nx
    xM, xm, yM, ym = analysis_coverage

    print(f"Visual coverage: {visual_coverage}, Sigmas: {sigmas}, NS: {ns}")
    print(f"Directories: {dirs}")

    #set device for torch
    torch.cuda.set_device(device)

    # Build paths
    pathdata = dirs[0] 
    pathsuite2p = pathdata + '/suite2p'

    deg_per_pix = abs(xM - xm) / nx
    sigmas_deg = np.trunc(2 * deg_per_pix * sigmas * 100) / 100

    print(f"Data path: {pathdata}")
    print(f"Suite2p path: {pathsuite2p}")

    # Load spike data
    if spks_path == 'None':
        print('Aligning data...')
        spks, spks_n, neuron_pos = lpn.loadSPKMesoscope(dirs[0], pathsuite2p, block_end, n_planes, nb_frames,
                                                        threshold=1.25, last=True, method='frame2ttl')
        neuron_pos = lpn.correctNeuronPos(neuron_pos, resolution)

    else:
        print(f'Loading spks file from {spks_path}')
        try:
            spks = np.load(spks_path)
            parent_dir = os.path.dirname(spks_path)
            neuron_pos = np.load(os.path.join(parent_dir, 'pos.npy'))
        except Exception as e:
            print(f"Error loading file: {e}")
            raise ValueError("Could not load spike data from specified path")

    print(f"Spike data shape: {spks.shape}")
    print(f"Neuron positions shape: {neuron_pos.shape}")

    # Compute response reliability and skewness
    print("Computing response reliability and skewness...")

    # Check if we have multiple trials for reliability calculation
    n_neurons = spks.shape[0]
    if n_trial2keep > 1:
        # Multiple trials - compute cross-trial reliability
        respcorr = au.repetability_trial3(spks, neuron_pos, plotting=False)
    else:
        # Single trial - skip reliability, set to default values
        print("Single trial detected - skipping cross-trial reliability calculation")
        respcorr = np.ones(n_neurons)  # No filtering based on reliability for single trial

    skewness = au.compute_skewness_neurons(spks, plotting=False)
    skewness = np.array(skewness)

    # Create filter mask
    if n_trial2keep > 1:
        filter_mask = np.logical_and(respcorr >= 0.2, skewness <= 20)
    else:
        # For single trial, only filter by skewness
        filter_mask = skewness <= 20

    print(f"Neurons passing filter: {np.sum(filter_mask)}/{n_neurons}")


    # Load wavelets
    parent_dir = os.path.dirname(movpath)
    print(f"Loading wavelets from {parent_dir}...")

    # First try: load pre-computed downsampled wavelets
    try:
        wavelets_downsampled = np.load(os.path.join(parent_dir, 'dwt_downsampled_videodata.npy'))
        w_r_downsampled = wavelets_downsampled[0]
        w_i_downsampled = wavelets_downsampled[1]
        w_c_downsampled = wavelets_downsampled[2]
        del wavelets_downsampled
        gc.collect()
        print("Loaded downsampled wavelets")
    except Exception as e:
        print(f"Downsampled wavelets not found: {e}")
    
    # Second try: load coarse wavelets
    try:
        print("Attempting to load coarse wavelets...")
        w_r_downsampled, w_i_downsampled, w_c_downsampled = lpn.coarseWavelet(parent_dir, False, nx0, ny0, nx, ny,
                                                                                n_theta, ns)
        print("Loaded coarse wavelets")
    except Exception as e:
        print(f"Error loading wavelets: {e}")
        
        # Third try: Check if downsampled video exists and generate wavelets
        downsampled_video_path = movpath[:-4] + '_downsampled.npy'
        if os.path.exists(downsampled_video_path):
            print(f"Found downsampled video at {downsampled_video_path}")
            print("Generating wavelet decomposition from downsampled video...")
            try:
                videodata = np.load(downsampled_video_path)
                print(f"Video data shape: {videodata.shape}")
                
                # Use optimized batched version for faster wavelet decomposition
                wg.waveletDecomposition_batched(videodata, [0, 1], sigmas, parent_dir, library_path=lib_path, device=device, batch_size=32)
                
                w_r_downsampled, w_i_downsampled, w_c_downsampled = lpn.coarseWavelet(parent_dir, False, nx0, ny0, nx, ny,
                                                                                        n_theta, ns)
                print("Completed wavelet decomposition from existing downsampled video")
            except Exception as e2:
                raise RuntimeError(f"Error in wavelet decomposition from downsampled video: {e2}")
        else:
            # Fourth try: Full pipeline - downsample video then decompose
            print("Attempting full video processing pipeline...")
            try:
                if (visual_coverage != analysis_coverage):
                    visual_coverage_arr = np.array(visual_coverage)
                    analysis_coverage_arr = np.array(analysis_coverage)
                    ratio_x = 1 - ((visual_coverage_arr[0] - visual_coverage_arr[1]) - (
                            analysis_coverage_arr[0] - analysis_coverage_arr[1])) / (
                                        visual_coverage_arr[0] - visual_coverage_arr[1])
                    ratio_y = 1 - ((visual_coverage_arr[2] - visual_coverage_arr[3]) - (
                            analysis_coverage_arr[2] - analysis_coverage_arr[3])) / (
                                        visual_coverage_arr[2] - visual_coverage_arr[3])
                else:
                    ratio_x = 1
                    ratio_y = 1
                
                print(f"Downsampling video: {movpath}")
                wg.downsample_video_binary(movpath, visual_coverage, analysis_coverage, shape=(ny, nx), chunk_size=1000,
                                        ratios=(ratio_x, ratio_y))
                videodata = np.load(movpath[:-4] + '_downsampled.npy')
                print(f"Downsampled video shape: {videodata.shape}")
                
                # Use optimized batched version for faster wavelet decomposition
                wg.waveletDecomposition_batched(videodata, [0, 1], sigmas, parent_dir, library_path=lib_path, batch_size=32)
                
                w_r_downsampled, w_i_downsampled, w_c_downsampled = lpn.coarseWavelet(parent_dir, False, nx0, ny0, nx, ny,
                                                                                        n_theta, ns)
                print("Completed full wavelet decomposition")
            except Exception as e3:
                raise RuntimeError(f"Error in full pipeline: {e3}")

    # Compute receptive fields using Pearson correlation
    print("Computing receptive fields...")
    print(f"w_c_downsampled shape: {w_c_downsampled.shape}")
    print(f"Expected: ({nb_frames}, {nx}, {ny}, {n_theta}, {ns})")

    # Use the actual number of frames we have
    n_frames_to_use = min(w_c_downsampled.shape[0], spks.shape[1])
    print(f"Using {n_frames_to_use} frames for RF calculation")

    print(spks[:, :n_frames_to_use].shape)

    rfs_gabor = au.PearsonCorrelationPinkNoise_batched( stim = w_c_downsampled[:n_frames_to_use].reshape(n_frames_to_use, -1), 
                                                resp = spks[:, :n_frames_to_use],
                                                neuron_pos= neuron_pos, 
                                                nx = nx, 
                                                ny = ny, 
                                                n_theta = n_theta,
                                                ns = ns, 
                                                visual_coverage = analysis_coverage, 
                                                screen_ratio = screen_ratio, 
                                                sigmas = sigmas_deg
                                                )

    # Plot retinotopy maps
    fig2, ax2 = plt.subplots(2, 2, figsize=(14, 12))
    maxes1 = rfs_gabor[2]
    plt.rcParams['axes.facecolor'] = 'none'

    m = ax2[0, 0].scatter(neuron_pos[:, 0], neuron_pos[:, 1], s=10, c=maxes1[0], cmap='jet', alpha=filter_mask)
    fig2.colorbar(m, ax=ax2[0, 0])
    ax2[0, 0].set_title('Azimuth Preference (deg)')
    ax2[0, 0].set_xlabel('X (um)')
    ax2[0, 0].set_ylabel('Y (um)')

    m = ax2[0, 1].scatter(neuron_pos[:, 0], neuron_pos[:, 1], s=10, c=maxes1[1], cmap='jet_r', alpha=filter_mask)
    fig2.colorbar(m, ax=ax2[0, 1])
    ax2[0, 1].set_title('Elevation Preference (deg)')
    ax2[0, 1].set_xlabel('X (um)')
    ax2[0, 1].set_ylabel('Y (um)')

    m = ax2[1, 0].scatter(neuron_pos[:, 0], neuron_pos[:, 1], s=10, c=maxes1[2], cmap='hsv', alpha=filter_mask)
    fig2.colorbar(m, ax=ax2[1, 0])
    ax2[1, 0].set_title('Orientation Preference (deg)')
    ax2[1, 0].set_xlabel('X (um)')
    ax2[1, 0].set_ylabel('Y (um)')

    m = ax2[1, 1].scatter(neuron_pos[:, 0], neuron_pos[:, 1], s=10, c=maxes1[3], cmap='coolwarm', alpha=filter_mask)
    fig2.colorbar(m, ax=ax2[1, 1])
    ax2[1, 1].set_title('Preferred Size (deg)')
    ax2[1, 1].set_xlabel('X (um)')
    ax2[1, 1].set_ylabel('Y (um)')

    plt.tight_layout()
    plt.show()

    # Save results
    print("Saving results...")
    save_path = dirs[0] + "/zebra/"
    np.save(os.path.join(save_path, 'correlation_matrix.npy'), rfs_gabor[0])
    np.save(os.path.join(save_path, 'maxes_indices.npy'), rfs_gabor[1])
    np.save(os.path.join(save_path, 'maxes_corrected.npy'), rfs_gabor[2])

    print("Analysis complete!")
    results = {
    'spks': spks,
    'neuron_pos': neuron_pos,
    'rfs_gabor': rfs_gabor,
    'filter_mask': filter_mask,
    'respcorr': respcorr,
    'skewness': skewness
    }


    save_path = dirs[0] + "/zebra/analysis_results.npy"
    np.save(save_path, results)

    # free memory
    del w_r_downsampled, w_i_downsampled, w_c_downsampled
    gc.collect()
    
    return results

if __name__ == "__main__":
    main()