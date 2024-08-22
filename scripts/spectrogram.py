import os
import toml
import logging
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import fftshift
from scipy.signal import get_window
from scipy.fftpack import fft
from scripts import data_processing as dpr
from scipy.signal import square, ShortTimeFFT

from scipy.signal.windows import gaussian


logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("pillow").setLevel(logging.WARNING)

with open(os.path.join(os.getcwd(), "config/spectrogram.toml")) as f:
    config = toml.load(f)

logger = logging.getLogger("__spectrogram__")



def compute_spectrogram(data, Fs):
    # Define the parameters for the spectrogram
    window_size = 256  # Window size for the FFT
    overlap = 128  # Overlap between windows
    
    # Window the signal
    window = np.hanning(window_size)
    windows = [data[i:i + window_size] * window for i in range(0, len(data) - window_size, window_size - overlap)]
    
    # Compute the FFT for each window
    spectrogram = [np.abs(np.fft.rfft(win)) ** 2 for win in windows]
    
    # Transpose the result to have time on the x-axis and frequency on the y-axis
    spectrogram = np.array(spectrogram).T
    
    # Plot the spectrogram
    frequencies = np.fft.rfftfreq(window_size, d=1.0 / Fs)
    time = np.arange(len(spectrogram[0])) * (window_size - overlap) / Fs
    
    return time, frequencies, spectrogram



def draw():
    path = config['dataset']['path']
    
    
    df = pd.read_csv(path, skiprows=config['signal']['behead'])
    # fig, axes = plt.subplots(6, 10, sharex=True, sharey=True)
    
    i = 0


    # processing subdivisions
    n_samples = 0
    if config['signal']['end_index']:
        data = df.loc[config['signal']['start_index']:config['signal']['end_index'], df.columns[config['signal']['column_index']]]
    else:
        data = df.loc[:, df.columns[config['signal']['column_index']]]
    
    f, t, Sxx = signal.spectrogram(data, 10000)
    clean_fft, clean_freqs = dpr.fast_fourier(data, config['signal']['sampling_frequency'])
    
    time, frequencies, spectrum = compute_spectrogram(data, Fs=10000)
    
    
    
    T_x, N = 1 / 10000, 600000  # 20 Hz sampling rate for 50 s signal
    t_x = np.arange(N) * T_x  # time indexes for signal
    f_i = 5e-3 * (t_x - t_x[N // 3]) ** 2 + 1  # varying frequency
    g_std = 12  # standard deviation for Gaussian window in samples
    win = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian wind.
    SFT = ShortTimeFFT(win, hop=2, fs=1 / T_x, mfft=800, scale_to='psd')
    Sx2 = SFT.spectrogram(data.values)  # calculate absolute square of STFT
    
    # Plot Spectrogram
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.specgram(Sxx, Fs=10000, cmap='rainbow')
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.pcolormesh(time, frequencies, 10 * np.log10(spectrum))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power spectral density (dB/Hz)")
    plt.ylim([0, 10000 / 2.])
    
    # Plot Frequency Profile
    plt.subplot(3, 1, 3)
    # plt.plot(freqs, 10 * np.log10(full_fft))
    plt.plot(clean_freqs, clean_fft)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [pV]')
    plt.title('Frequency Profile')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
    
    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    
    ax1.set_title(rf"Spectrogram ({SFT.m_num * SFT.T:g}$\,s$ Gaussian " +
                  
                  rf"window, $\sigma_t={g_std * SFT.T:g}\,$s)")
    
    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
                   
                   rf"$\Delta t = {SFT.delta_t:g}\,$s)",
            
            ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                   
                   rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
            
            xlim=(t_lo, t_hi))
    
    Sx_dB = 10 * np.log10(np.fmax(Sx2, 1e-4))  # limit range to -40 dB
    
    im1 = ax1.imshow(Sx_dB, origin='lower', aspect='auto',
                     
                     extent=SFT.extent(N), cmap='magma')
    
    ax1.plot(t_x, f_i, 'g--', alpha=.5, label='$f_i(t)$')
    
    fig1.colorbar(im1, label='Power Spectral Density ' +
                             
                             r"$20\,\log_{10}|S_x(t, f)|$ in dB")
    
    # Shade areas where window slices stick out to the side:
    
    for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                     
                     (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
        ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.3)
    
    for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line
        
        ax1.axvline(t_, color='c', linestyle='--', alpha=0.5)
    
    ax1.legend()
    
    fig1.tight_layout()
    
    plt.show()