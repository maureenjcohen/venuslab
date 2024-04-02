""" Spectral analysis"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from venusdyn import time_series

# %%
def timeseries_transform(plobject, key='vitw', fsize=14, plot_transform=True,
                         frequnit=1./20,
                         coords=[(16,86,48),(22,86,48),(30,86,48)],
                         trange=[1777,1877], save=False, saveformat='png',
                         savename='fourier_transform.png'):
    """ Perform Fourier transform of a time series 
        Plot power spectral density against frequency

        Input arg frequnit sets the units of the frequency axis (x-ax9s)
        Default is 1/20 because the aoa_surface dataset has time units of
        20 outputs per Venus day """

    series_list, coords_list = time_series(plobject, key=key, 
                coords=coords, 
                ptitle='filler', ylab='filler', 
                unit='filler', plot=False, trange=trange, 
                tunit='filler',
                fsize=14, save=False, saveformat='png')
    if plot_transform==True: 
        fig, ax = plt.subplots(figsize=(8,6))
        colors=['tab:blue','tab:green','tab:orange']
        for ind, item in enumerate(series_list):
            fft = sp.fftpack.fft(item)
            psd = np.abs(fft)**2
            freqs = sp.fft.fftfreq(len(fft),d=frequnit)
            i = freqs > 0

            print('Plotting item ' + str(ind))
            plt.plot(freqs[i], psd[i],
                    color=colors[ind],
                    label=f'{int(coords_list[ind][1])}$^\circ$ lat, {int(coords_list[ind][2])}$^\circ$ lon, {int(coords_list[ind][0])} km')
        ax.set_title('Vertical wind wave frequencies', fontsize=fsize+2)
        ax.set_xlabel('Frequency / Venus day$^{-1}$', fontsize=fsize)
        ax.set_ylabel(r'Power spectral density / m$^2$s$^{-2}$Hz$^{-1}$', fontsize=fsize)
        ax.set_xlim(0,5)
        plt.legend()
        if save==True:
            plt.savefig(savename, format=saveformat, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# %%
def bandpass(datarray, frequnit=1, plot=True):
    """ Perform Fourier transform of 1-D input array
        Plot PSDs
        Remove all frequencies except background """
    fft = sp.fftpack.fft(datarray)
    psd = np.abs(fft)**2
    freqs = sp.fft.fftfreq(len(fft),d=frequnit)
    i = freqs > 0
    
    fft_pass = fft.copy()
    fft_pass[np.abs(freqs)> 0.] = 0
    cleaned = np.real(sp.fftpack.ifft(fft_pass))

    if plot==True:
        fig, ax = plt.subplots(figsize=(8,6))   
        ax.plot(freqs[i], psd[i])
        ax.set_title('Fourier transform')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power spectral density')
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(8,6))
        ax2.plot(datarray)
        ax2.plot(cleaned)
        plt.show()

    return cleaned

# %%
