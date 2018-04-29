
from numpy.fft import fft,ifft
import numpy
from matplotlib import pyplot as plt

def shift(x,n=0): 
    vec=0*x  
    vec[n]=1
    vecft=fft(vec)
    xft=fft(x)
    return numpy.real(ifft(xft*vecft))

if __name__=='__main__':
    x=numpy.arange(-15,15,0.1)
    sigma=2
    y=numpy.exp(-0.5*x**2/sigma**2)
    yshift=shift(y,y.size/2)
    
    plt.ion()
    plt.plot(x,y)
plt.plot(x,yshift)


def corrF(x,y):
    assert(x.size==y.size)  
    xft=fft(x)
    yft=fft(y)
    yftconj=numpy.conj(yft)
    return numpy.real(ifft(xft*yftconj))

if __name__=='__main__':
        x=numpy.arange(-20,20,0.1)
        sigma=2
        y=numpy.exp(-0.5*x**2/sigma**2)
        
        ycorr=corrF(y,y)
        plt.plot(x,ycorr)
plt.show()

def shift(x,n=0):
    vec=0*x  
    vec[n]=1
    vecft=fft(vec)
    xft=fft(x)
    return numpy.real(ifft(xft*vecft))



def corrF(x,y):
    assert(x.size==y.size)  
    xft=fft(x)
    yft=fft(y)
    yftconj=numpy.conj(yft)
    return numpy.real(ifft(xft*yftconj))

if __name__=='__main__':
        x=numpy.arange(-15,15,0.1)
        sigma=2
        y=numpy.exp(-0.5*x**2/sigma**2)
        
        ycorr=corrF(y,y)
        yshift=myshift(y,y.size/4)
        yshiftcorr=corrF(yshift,yshift)
        mean_error=numpy.mean(numpy.abs(ycorr-yshiftcorr))
        print 'mean difference between the two correlation functions is ' + repr(mean_error)
        plt.plot(x,ycorr)
        plt.plot(x,yshiftcorr)        
plt.show()


def convolut(x,y):
    assert(x.size==y.size)  
    xx=numpy.zeros(2*x.size)
    xx[0:x.size]=x

    yy=numpy.zeros(2*y.size)
    yy[0:y.size]=y
    xxft=fft(xx)
    yyft=fft(yy)
    vec=numpy.real(ifft(xxft*yyft))
    return vec[0:x.size]

if __name__=='__main__':
    x=numpy.arange(-15,15,0.1)
    sigma=2
    y=numpy.exp(-0.5*x**2/sigma**2)
    y=y/y.sum()

    yconv=convolut(y,y)
    plt.plot(x,y)
    plt.plot(x,yconv)
plt.show()