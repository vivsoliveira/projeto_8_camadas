import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift
from suaBibSignal import signalMeu
import sounddevice as sd

def main():
    file_path = './casca_de_bala.wav' 
    audio, sample_rate = sf.read(file_path)
    audio = audio[:, 0]  
    audio /= np.max(np.abs(audio))
    duration = len(audio) / sample_rate
    time = np.linspace(0, duration, len(audio))
    eu = signalMeu()
    print(sample_rate)
    def low_pass_filter(audio, cutoff_hz, sample_rate):
        nyq_rate = sample_rate / 2.
        width = 5.0 / nyq_rate  # transition width
        ripple_db = 60.0  # attenuation in the stop band
        N, beta = signal.kaiserord(ripple_db, width)
        taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))
        filtered_signal = signal.lfilter(taps, 1.0, audio)
        return filtered_signal

    # Apply low-pass filter
    cutoff_frequency = 4000.0
    filtered_audio = low_pass_filter(audio, cutoff_frequency, sample_rate)

    # Normalize filtered audio
    filtered_audio /= np.max(np.abs(filtered_audio))


    def am_modulate(signal, carrier_freq, sample_rate):
        t = np.arange(len(signal)) / sample_rate
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        return signal * carrier

    carrier_frequency = 14000  # Carrier frequency for AM modulation
    modulated_audio = am_modulate(filtered_audio, carrier_frequency, sample_rate)

    # Normalize modulated audio
    modulated_audio /= np.max(np.abs(modulated_audio))

    def am_demodulate(modulated_signal, carrier_freq, sample_rate):
        t = np.arange(len(modulated_signal)) / sample_rate
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        demodulated = modulated_signal * carrier
        return low_pass_filter(demodulated, cutoff_frequency, sample_rate)

    demodulated_audio = am_demodulate(modulated_audio, carrier_frequency, sample_rate)

    # Normalize demodulated audio
    demodulated_audio /= np.max(np.abs(demodulated_audio))
    sd.play(audio, sample_rate)
    sd.wait()
    sd.play(modulated_audio, sample_rate)
    sd.wait()
    sd.play(demodulated_audio, sample_rate)
    sd.wait()
    

    demod_fil = low_pass_filter(demodulated_audio, cutoff_frequency, sample_rate)

    # Plot all graphs in one figure
    plt.figure(figsize=(12, 10))

    # Original Audio
    plt.subplot(2, 1, 1)
    plt.plot(time, audio)
    plt.title('Original Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Filtered Audio
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_audio)
    plt.title('Filtered Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Filtered Audio FFT
    eu.plotFFT(filtered_audio, 48000)
    plt.title('Filtered Audio FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Modulated Audio
    plt.plot(time, modulated_audio)
    plt.title('AM Modulated Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Modulated Audio FFT
    eu.plotFFT(modulated_audio, 48000)
    plt.title('AM Modulated Audio FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Demodulated Audio
    plt.plot(time, demodulated_audio)
    plt.title('AM Demodulated Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Demodulated Audio FFT
    eu.plotFFT(demodulated_audio, 48000)
    plt.title('AM Demodulated Audio FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Demolutado e filtrado
    eu.plotFFT(demod_fil, 48000)
    plt.title('Demodulado filtrado')
    plt.xlabel('Frequencia [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()