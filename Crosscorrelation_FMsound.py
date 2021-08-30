#2021/8/30 22:00
#kanta crate CrossCorrelation_FMsound
import math
from os import fsdecode
from matplotlib import scale 
import numpy as np
from numpy.core.fromnumeric import transpose
from numpy.lib import imag
from scipy import signal
from scipy.signal import chirp
from scipy.fftpack import fft2
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy import fftpack
import cmath

from scipy.signal.ltisys import freqresp

#実装用パラメータ設定
def initilize(): 
    Boxnum=32768
    amp=32767. / 2.#音圧
    fr=192000 #サンプリング周波数
    fpgafreq=1000000 #点数？
    dur=0.010  #duration
    Fstart=80000#開始周波数
    FEnd=40000#終端周波数
    BatCallConstant=0.005#パラメータ
    delay=0
    
    sig2=[0]*Boxnum
    crosscor=[0]*Boxnum
    return Fstart,FEnd,BatCallConstant,amp,fr,fpgafreq,dur,Boxnum,delay,sig2,crosscor

#コウモリFM型信号作成
def makeFMsound(fStart,fEnd,BatCallConstant,amp,fs,dur):
    pi =np.pi
    nframes=int(dur*fs+1)
    arg= (BatCallConstant*fEnd)/fStart
    call=[]
    fStart=fStart/100
    fEnd=fEnd/100
    sum=0
    for i in range(nframes):
        t = float(i)/fs*100
        call.append(amp*np.sin(2.*pi*((fStart/(fStart-BatCallConstant*fEnd))*((fStart-fEnd)*np.float_power(arg, t)/math.log(arg)+(1-BatCallConstant)*fEnd*t))))
    
    print("nframeの個数は",nframes)

    return call
#STFTグラフ表示
def STFT(data,fs):
    arr_data = np.array(data)#リストを配列に直す
    f, t, spectrogram = signal.spectrogram(arr_data, fs=fs,nfft=1024,nperseg=256)
    return t,f,spectrogram
#FFTしてグラフ表示
def FFT(data, fs,dur):
    freq_data = fftpack.fft(data)
    fft_fre = fftpack.fftfreq(n=int(dur*fs+1), d=1/fs)
    return freq_data,fft_fre

def plot_FFT(call_data,call_fre):
    plt.plot(call_fre,abs(call_data))
    plt.xlim(0,90000)
    plt.title("DownFM 80-40 kHz 10ms FFT")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [a.u.]")
    plt.pause(1)
    plt.savefig("plot_fft_10ms")
#soundの振幅波形とSTFTを表示
def plot_signal(fStart,fEnd,BatCallConstant,amp,fs,dur):
    fig=plt.figure()
     #callのSTFT
    BatCallConstant=0.005
    sig1=makeFMsound(fStart,fEnd,BatCallConstant,amp,fs,dur)
    t,f,spectrogram=STFT(sig1,fs)
    ax3=fig.add_subplot(2,2,1)
    ax3.pcolormesh(t,f,np.log(spectrogram))
    ax3.set_xlabel("Time [sec]")
    ax3.set_ylabel("Freq [Hz]")
    ax3.set_ylim(30000,90000)
    ax3.set_title("a=0.005")
    #callのSTFT
    BatCallConstant=0.010
    sig1=makeFMsound(fStart,fEnd,BatCallConstant,amp,fs,dur)
    t,f,spectrogram=STFT(sig1,fs)
    ax3=fig.add_subplot(2,2,2)
    ax3.pcolormesh(t,f,np.log(spectrogram))
    ax3.set_xlabel("Time [sec]")
    ax3.set_ylabel("Freq [Hz]")
    ax3.set_ylim(30000,90000)
    ax3.set_title("a=0.010")
    #callのSTFT
    BatCallConstant=0.015
    sig1=makeFMsound(fStart,fEnd,BatCallConstant,amp,fs,dur)
    t,f,spectrogram=STFT(sig1,fs)
    ax3=fig.add_subplot(2,2,3)
    ax3.pcolormesh(t,f,np.log(spectrogram))
    ax3.set_xlabel("Time [sec]")
    ax3.set_ylabel("Freq [Hz]")
    ax3.set_ylim(30000,90000)
    ax3.set_title("a=0.015")
    #callのSTFT
    BatCallConstant=0.020
    sig1=makeFMsound(fStart,fEnd,BatCallConstant,amp,fs,dur)
    t,f,spectrogram=STFT(sig1,fs)
    ax3=fig.add_subplot(2,2,4)
    ax3.pcolormesh(t,f,np.log(spectrogram))
    ax3.set_xlabel("Time [sec]")
    ax3.set_ylabel("Freq [Hz]")
    ax3.set_ylim(30000,90000)
    ax3.set_title("a=0.020")
    fig.tight_layout()
    plt.savefig("plot_signal_10ms.png")
"""
def plot_signal(sig1,fs,dur):
    fig=plt.figure()

    #callの生波形
    ax1=fig.add_subplot(3,1,1)
    ax1.plot(sig1)
    ax1.set_title("DownFM 80-40 kHz 10ms")
    ax1.set_xlabel("Time(data)")
    ax1.set_ylabel("音圧 [V]")
    #callのFFTした図
    call_data,call_fre=FFT(sig1,fs,dur)
    ax2=fig.add_subplot(3,1,2)
    ax2.plot(call_fre,abs(call_data))
    ax2.set_xlim(0,90000)
    ax2.set_title("DownFM 80-40 kHz 10ms FFT")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [a.u.]")
    #callのSTFT
    t,f,spectrogram=STFT(sig1,fs)
    ax3=fig.add_subplot(3,1,3)
    ax3.pcolormesh(t,f,np.log(spectrogram))
    ax3.set_xlabel("Time [sec]")
    ax3.set_ylabel("Freq [Hz]")
    ax3.set_ylim(30000,90000)
    fig.tight_layout()
    plt.savefig("plot_signal_10ms.png")
"""
#callの信号をFFT(complex)
def call_FFT(sig1,Boxnum,fpgafreq,dur):
    sig1_1=[]
    sig1_1=np.append(sig1_1,sig1)
    sig1_1=np.append(sig1_1,[0]*(Boxnum-(int(dur*fpgafreq+1))))
    sig1_ft=fftpack.fft(sig1_1)
    fft_fre = fftpack.fftfreq(n=Boxnum, d=1/fpgafreq)
    return sig1_ft,fft_fre,sig1_1
#sigの信号をFFT(complex)
def sig_FFT(sig1,delay,dur,fs,Boxnum):
    sig2=[]
    sig2=np.append(sig2,[0]*delay)
    sig2=np.append(sig2,sig1)
    sig2=np.append(sig2,[0]*(Boxnum-delay-(int(dur*fs+1))))
    sig2_ft=fftpack.fft(sig2)
    fft_fre = fftpack.fftfreq(n=Boxnum, d=1/fpgafreq)
    return sig2_ft,fft_fre,sig2

def multiplySpecs(sig1_FT,sig2_FT,Boxnum):
    mix_FT=[]
    hil_FT=[]
    for i in range(Boxnum):
        mix_FT=np.append(mix_FT,np.conjugate(sig1_FT[i]*sig2_FT[i]))
        if i<=Boxnum/2:
            hil_FT=np.append(hil_FT,complex(np.imag(mix_FT[i]),np.real(-mix_FT[i])))
        else:
            hil_FT=np.append(hil_FT,complex(np.imag(-mix_FT[i]),np.real(mix_FT[i])))
    return mix_FT,hil_FT

def calculateEnvelope(mix_FT,hil_FT,Boxnum,fpgafreq):
    CrossCor=[]
    mix_FT_inv=np.fft.ifft(mix_FT)
    hil_FT_inv=np.fft.ifft(hil_FT)
    for i in range(Boxnum):
        CrossCor=np.append(CrossCor,np.sqrt(np.real(mix_FT_inv[i])*np.real(mix_FT_inv[i])+np.real(hil_FT_inv[i])*np.real(hil_FT_inv[i])))
    #CrossCor=np.roll(CrossCor,-(Boxnum//2))
    time = np.arange(Boxnum) / float(fpgafreq)
    return CrossCor,time

def plot_cross_only(CrossCor,Boxnum):
    data = np.arange(Boxnum) - int(Boxnum/2)
    plt.plot(data,CrossCor)
    plt.title("Crosscor")
    plt.xlabel("data")
    plt.ylabel("Amplitude")
    plt.pause(0.1)
    plt.savefig("plot_corr_15ms.png")
    

def plot_cross(CrossCor,time,fs,sig1_1,sig2):
    fig=plt.figure()
    ax1=fig.add_subplot(3,1,1)
    t,f,spectrogram=STFT(sig1_1,fs)
    ax1.pcolormesh(t,f,np.log(spectrogram))
    ax1.set_xlabel("Time [sec]")
    ax1.set_ylabel("Freq [Hz]")
    ax1.set_ylim(30000,90000)
   
    ax2=fig.add_subplot(3,1,2)
    t,f,spectrogram=STFT(sig2,fs)
    ax2.pcolormesh(t,f,np.log(spectrogram))
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Freq [Hz]")
    ax2.set_ylim(30000,90000)
   
    ax3=fig.add_subplot(3,1,3)
    ax3.plot(CrossCor)
    ax3.set_title("Crosscor")
    ax3.set_xlabel("data")
    ax3.set_ylabel("Amplitude")

    fig.tight_layout()
    plt.savefig("plot_cross_10ms.png")

#python のcrosscor関数を使ったver
def CrossCorfunction_py(sig1_1,sig2):
     corr=np.correlate(sig1_1,sig2,"full")
     return corr

def plot_cross_funcpy(corr,fs,sig1_1,sig2):
    fig=plt.figure()
    ax1=fig.add_subplot(3,1,1)
    t,f,spectrogram=STFT(sig1_1,fs)
    ax1.pcolormesh(t,f,np.log(spectrogram))
    ax1.set_xlabel("Time [sec]")
    ax1.set_ylabel("Freq [Hz]")
    ax1.set_ylim(30000,90000)
   
    ax2=fig.add_subplot(3,1,2)
    t,f,spectrogram=STFT(sig2,fs)
    ax2.pcolormesh(t,f,np.log(spectrogram))
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Freq [Hz]")
    ax2.set_ylim(30000,90000)
   
    ax3=fig.add_subplot(3,1,3)
    ax3.plot(corr)
    ax3.set_title("Crosscor")
    ax3.set_xlabel("time ")
    ax3.set_ylabel("Amplitude")
  

    fig.tight_layout()
    plt.savefig("plot_corr_funcpy%d.png",i)






if __name__ == '__main__':
    """
    FStart,FEnd,BatCallConstant,amp,fr,fpgafreq,dur,Boxnum,delay,sig2,crosscor=initilize()
    sig1=makeFMsound( FStart,FEnd,BatCallConstant,amp,fpgafreq,dur)
    plot_signal(sig1,fpgafreq,dur)
    sig1_FT,fft_fre,sig1_1=call_FFT(sig1,Boxnum,fpgafreq,dur)
    sig2_FT,fft_fre,sig2=sig_FFT(sig1,delay,dur,fpgafreq,Boxnum)
    mix_FT,hil_FT=multiplySpecs(sig1_FT,sig2_FT,Boxnum)
    CrossCor,time=calculateEnvelope(mix_FT,hil_FT,Boxnum,fpgafreq)
    plot_cross(CrossCor,time,fpgafreq,sig1_1,sig2)
    #plot_cross_only(CrossCor)
   """
    
    FStart,FEnd,BatCallConstant,amp,fr,fpgafreq,dur,Boxnum,delay,sig2,crosscor=initilize()
    plot_signal(FStart,FEnd,BatCallConstant,amp,fpgafreq,dur)
    """  
     for BatCallConstant in range(5,25,5):
         sig1=makeFMsound( FStart,FEnd,BatCallConstant*0.01,amp,fpgafreq,dur)
         call_data,call_fre,sig1_1=call_FFT(sig1,Boxnum,fpgafreq,dur)
         plot_FFT(call_data,call_fre)
      
    
    for BatCallConstant in range(5,30,5):
         
        sig1=makeFMsound( FStart,FEnd,BatCallConstant*0.01,amp,fpgafreq,dur)
      #  plot_signal(sig1,fpgafreq,dur)
        sig1_FT,fft_fre,sig1_1=call_FFT(sig1,Boxnum,fpgafreq,dur)
        sig2_FT,fft_fre,sig2=sig_FFT(sig1,delay,dur,fpgafreq,Boxnum)
        mix_FT,hil_FT=multiplySpecs(sig1_FT,sig2_FT,Boxnum)
        CrossCor,time=calculateEnvelope(mix_FT,hil_FT,Boxnum,fpgafreq)
        #plot_cross(CrossCor,time,fpgafreq,sig1_1,sig2)
        plot_cross_only(CrossCor,Boxnum)
    
      
    """  
    
    """
    corr=CrossCorfunction_py(sig1_1,sig2)
    plot_cross_funcpy(corr,fpgafreq,sig1_1,sig2)
    """

   # crosscor_func(sig1,delay,dur,fpgafreq,Boxnum)


    """
    
    ##デバック用（mixFT_real_fft）
    mix_FT_str='\n'.join(map(str, mix_FT))
    f=open('mix_FT.txt','w')
    f.writelines(mix_FT_str)
    f.close()

    ##デバック用（hilFT_real_fft）
    hil_FT_str='\n'.join(map(str, hil_FT))
    f=open('hil_FT.txt','w')
    f.writelines(hil_FT_str)
    f.close()

    
    
    ##デバック用（call_fft）
    sig1_FT_str='\n'.join(map(str, sig1_FT))
    f=open('sig1_FT.txt','w')
    f.writelines(sig1_FT_str)
    f.close()
    

    
    
    ##デバック用（sig_fft）
    sig2_FT_str='\n'.join(map(str, sig2_FT))
    f=open('sig2_FT.txt','w')
    f.writelines(sig2_FT_str)
    f.close()
    

    


    
    
    ##デバック用（call_fft）
    sig1_FT_str='\n'.join(map(str, sig1_FT))
    f=open('sig_FT.txt','x')
    f.writelines(sig1_FT_str)
    f.close()
    """