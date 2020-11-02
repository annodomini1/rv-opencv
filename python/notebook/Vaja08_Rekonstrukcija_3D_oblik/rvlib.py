# -*- coding: utf-8 -*-
'''
Created on Mon Mar 16 08:47:50 2015

@author: Žiga Špiclin

RVLIB: knjižnica funkcij iz laboratorijskih vaj
       pri predmetu Robotski vid
'''
import numpy as np
import PIL.Image as im
import matplotlib.pyplot as plt
import matplotlib.cm as cm # uvozi barvne lestvice
from scipy.ndimage import convolve
from scipy.interpolate import interpn

def loadImageRaw(iPath, iSize, iFormat):
    '''
    Naloži sliko iz raw datoteke
    
    Parameters
    ----------
    iPath : str 
        Pot do datoteke
    iSize : tuple 
        Velikost slike
    iFormat : str
        Tip vhodnih podatkov
    
    Returns
    ---------
    oImage : numpy array
        Izhodna slika
    
    
    '''
    
    oImage = np.fromfile(iPath, dtype=iFormat) # nalozi raw datoteko
    oImage = np.reshape(oImage, iSize) # uredi v matriko
    
    return oImage


def showImage(iImage, iTitle=''):
    '''
    Prikaže sliko iImage in jo naslovi z iTitle
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika 
    iTitle : str 
        Naslov za sliko
    
    Returns
    ---------
    Nothing
    
    
    '''
    fig = plt.figure() # odpri novo prikazno okno
    ax = fig.add_subplot(111)
    if iImage.ndim == 3:
        iImage = np.transpose(iImage,[1,2,0])

    ax.imshow(iImage, cmap = cm.Greys_r) # prikazi sliko v novem oknu
    ax.set_title(iTitle) # nastavi naslov slike
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()

def saveImageRaw(iImage, iPath, iFormat):
    '''
    Shrani sliko na disk
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika za shranjevanje
    iPath : str
        Pot in ime datoteke, v katero želimo sliko shraniti
    iFormat : str
        Tip podatkov v matriki slike
    
    Returns
    ---------
    Nothing
    '''
    iImage = iImage.astype(iFormat)
    iImage.tofile(iPath) # zapisi v datoteko


def loadImage(iPath):
    '''
    Naloži sliko v standardnih formatih (bmp, jpg, png, tif, gif, idr.)
    in jo vrni kot matriko
    
    Parameters
    ----------
    iPath - str
        Pot do slike skupaj z imenom
        
    Returns
    ----------
    oImage - numpy.ndarray
        Vrnjena matrična predstavitev slike
    '''
    oImage = np.array(im.open(iPath))
    if oImage.ndim == 3:
        oImage = np.transpose(oImage,[2,0,1])
    elif oImage.ndim == 2:
        oImage = np.transpose(oImage,[1,0])   
    return oImage


def saveImage(iPath, iImage, iFormat):
    '''
    Shrani sliko v standardnem formatu (bmp, jpg, png, tif, gif, idr.)
    
    Parameters
    ----------
    iPath : str
        Pot do slike z željenim imenom slike
    iImage : numpy.ndarray
        Matrična predstavitev slike
    iFormat : str
        Željena končnica za sliko (npr. 'bmp')
    
    Returns
    ---------
    Nothing

    '''
    if iImage.ndim == 3:
        iImage = np.transpose(iImage,[1,2,0])
    elif iImage.ndim ==2:
        iImage = np.transpose(iImage,[1,0])     
    img = im.fromarray(iImage) # ustvari slikovni objekt iz matrike
    img.save(iPath.split('.')[0] + '.' + iFormat)


def drawLine(iImage, iValue, x1, y1, x2, y2):
    ''' Narisi digitalno daljico v sliko

        Parameters
        ----------
        iImage : numpy.ndarray
            Vhodna slika
        iValue : tuple, int
            Vrednost za vrisavanje (barva daljice).
            Uporabi tuple treh elementov za barvno sliko in int za sivinsko sliko
        x1 : int
            Začetna x koordinata daljice
        y1 : int
            Začetna y koordinata daljice
        x2 : int
            Končna x koordinata daljice
        y2 : int
            Končna y koordinata daljice
    '''    
    
    oImage = iImage    
    
    if iImage.ndim == 3:
        assert type(iValue) == tuple, 'Za barvno sliko bi paramter iValue moral biti tuple treh elementov'
        for rgb in range(3):
            drawLine(iImage[rgb,:,:], iValue[rgb], x1, y1, x2, y2)
    
    elif iImage.ndim == 2:
        assert type(iValue) == int, 'Za sivinsko sliko bi paramter iValue moral biti int'
    
        dx = np.abs(x2 - x1)
        dy = np.abs(y2 - y1)
        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1
        napaka = dx - dy
     
        x = x1
        y = y1
        
        while True:
            oImage[y-1, x-1] = iValue
            if x == x2 and y == y2:
                break
            e2 = 2*napaka
            if e2 > -dy:
                napaka = napaka - dy
                x = x + sx
            if e2 < dx:
                napaka = napaka + dx
                y = y + sy
    
    return oImage
    
    
def colorToGray(iImage):
    '''
    Pretvori barvno sliko v sivinsko.
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna barvna slika
        
    Returns
    -------
    oImage : numpy.ndarray
        Sivinska slika
    '''
    dtype = iImage.dtype
    r = iImage[0,:,:].astype('float')
    g = iImage[1,:,:].astype('float')
    b = iImage[2,:,:].astype('float')
    
    return (r*0.299 + g*0.587 + b*0.114).astype(dtype)
    
    
def computeHistogram(iImage, iNumBins, iRange=[], iDisplay=False, iTitle=''):
    '''
    Izracunaj histogram sivinske slike
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna slika, katere histogram želimo izračunati

    iNumBins : int
        Število predalov histograma
        
    iRange : tuple, list
        Minimalna in maksimalna sivinska vrednost 

    iDisplay : bool
        Vklopi/izklopi prikaz histograma v novem oknu

    iTitle : str
        Naslov prikaznega okna
        
    Returns
    -------
    oHist : numpy.ndarray
        Histogram sivinske slike
    oEdges: numpy.ndarray
        Robovi predalov histograma
    '''    
    iImage = np.asarray(iImage)
    iRange = np.asarray(iRange)
    if iRange.size == 2:
        iMin, iMax = iRange
    else:
        iMin, iMax = np.min(iImage), np.max(iImage)
    oEdges = np.linspace(iMin, iMax+1, iNumBins+1)
    oHist = np.zeros([iNumBins,])
    for i in range(iNumBins):
        idx = np.where((iImage >= oEdges[i]) * (iImage < oEdges[i+1]))
        if idx[0].size > 0:
            oHist[i] = idx[0].size
    if iDisplay:
        plt.figure()
        plt.bar(oEdges[:-1], oHist)
        plt.suptitle(iTitle)

    return oHist, oEdges
    
    
def computeContrast(iImages):
    '''
    Izracunaj kontrast slik
    
    Parameters
    ---------
    iImages : list of numpy.ndarray
        Vhodne slike, na katerih želimo izračunati kontrast
        
    Returns : list
        Seznam kontrastov za vsako vhodno sliko
    '''
    oM = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        fmin = np.percentile(iImages[i].flatten(),5)
        fmax = np.percentile(iImages[i].flatten(),95)
        oM[i] = (fmax - fmin)/(fmax + fmin)
    return oM
    
    
def computeEffDynRange(iImages):
    '''
    Izracunaj efektivno dinamicno obmocje
    
    Parameters
    ----------
    iImages : numpy.ndarray
        Vhodne slike
        
    Returns
    --------
    oEDR : float
        Vrednost efektivnega dinamicnega obmocja
    '''
    L = np.zeros((len(iImages,)))
    sig = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        L[i] = np.mean(iImages[i].flatten())
        sig[i] = np.std(iImages[i].flatten())
    oEDR = np.log2((L.max() - L.min())/sig.mean())
    return oEDR
    

def computeSNR(iImage1, iImage2):
    '''
    Vrne razmerje signal/sum
    
    Paramters
    ---------
    iImage1, iImage2 : np.ndarray
        Sliki področij zanimanja, med katerima računamo SNR
        
    Returns
    ---------
    oSNR : float
        Vrednost razmerja signal/sum
    '''
    mu1 = np.mean(iImage1.flatten())
    mu2 = np.mean(iImage2.flatten())
    
    sig1 = np.std(iImage1.flatten())
    sig2 = np.std(iImage2.flatten())
    
    oSNR = np.abs(mu1 - mu2)/np.sqrt(sig1**2 + sig2**2)
            
    return oSNR
    
    
def scaleImage( iImage, iSlopeA, iIntersectionB ):
    '''
    Linearna sivinska preslikava y = a*x + b
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iSlopeA : float
        Linearni koeficient (a) v sivinski preslikavi
        
    iIntersectionB : float
        Konstantna vrednost (b) v sivinski preslikavi
        
    Returns
    --------
    oImage : numpy.ndarray
        Linearno preslikava sivinska slika
    '''    
    iImageType = iImage.dtype
    iImage = np.array( iImage, dtype='float' )
    oImage = iSlopeA * iImage + iIntersectionB
    # zaokrozevanje vrednosti
    if iImageType.kind in ('u','i'):
        oImage[oImage<np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        oImage[oImage>np.iinfo(iImageType).max] = np.iinfo(iImageType).max
    return np.array( oImage, dtype=iImageType )
    
    
def windowImage( iImage, iCenter, iWidth ):
    '''
    Linearno oknjenje y = (Ls-1)/w*(x-(c-w/2)
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iCenter : float
        Sivinska vrednost, ki določa položaj centra okna
        
    iWidth : float
        Širina okna, ki določa razpon linearno preslikavnih vrednosti
        
    Returns
    --------
    oImage : numpy.ndarray
        Oknjena sivinska slika
    '''     
    iImageType = iImage.dtype
    if iImageType.kind in ('u','i'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max( iImage )
        iMinValue = np.max( iImage )
        iRange = iMaxValue - iMinValue
    
    iSlopeA = iRange / float(iWidth)
    iInterceptB = - iSlopeA * ( float(iCenter) - iWidth / 2.0 )
    
    return scaleImage( iImage, iSlopeA, iInterceptB )


def thresholdImage( iImage, iThreshold ):
    '''
    Upragovljanje y = x > t
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iThreshold : float
        Sivinska vrednost, ki določa prag
        
    Returns
    --------
    oImage : numpy.ndarray
        Upragovljena binarna slika
    '''         
    iImage = np.asarray( iImage )
    oImage = 255 * np.array(iImage>iThreshold, dtype='uint8')    
    return oImage    
    
    
def gammaImage( iImage, iGamma ):
    '''
    Upragovljanje y = (Ls-1)(x/(Lr-1))^gamma
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika

    iGamma : float
        Vrednost gama
        
    Returns
    --------
    oImage : numpy.ndarray
        Gama preslikana slika
    '''     
    iImage = np.asarray( iImage )
    iImageType = iImage.dtype
    iImage = np.array( iImage, dtype='float' )
    # preberi mejne vrednosti in obmocje vrednosti
    if iImageType.kind in ('u','i'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max( iImage )
        iMinValue = np.max( iImage )
        iRange = iMaxValue - iMinValue
    # izvedi gamma preslikavo
    iImage = (iImage - iMinValue) / float(iRange)
    oImage = iImage ** iGamma        
    oImage = float(iRange) * oImage + iMinValue
    # zaokrozevanje vrednosti
    if iImageType.kind in ('u','i'):
        oImage[oImage<np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        oImage[oImage>np.iinfo(iImageType).max] = np.iinfo(iImageType).max
    # vrni sliko v originalnem formatu
    return np.array( oImage, dtype=iImageType )
    

def convertImageColorSpace( iImage, iConversionType ):
    '''
    Pretvorba barvne slike med barvnima prostoroma RGB in HSV
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna RGB ali HSV slika

    iConversionType : str
        'RGBtoHSV' ali 'HSVtoRGB'
        
    Returns
    --------
    oImage : numpy.ndarray
        Preslikana RGB ali HSV slika
    '''       
    iImage = np.asarray( iImage )
    colIdx = [iImage.shape[i] == 3 for i in range(len(iImage.shape))]
    iImage = np.array( iImage, dtype='float' )
    
    if iConversionType == 'RGBtoHSV':
        if colIdx.index( True ) == 0:
            r = iImage[0,:,:]; g = iImage[1,:,:];  b = iImage[2,:,:];
        elif colIdx.index( True ) == 1:
            r = iImage[:,0,:]; g = iImage[:,1,:]; b = iImage[:,2,:];
        elif colIdx.index( True ) == 2:
            r = iImage[:,:,0]; g = iImage[:,:,1]; b = iImage[:,:,2];
        
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
    
        h = np.zeros_like( r )
        s = np.zeros_like( r )
        v = np.zeros_like( r )
    
        Cmax = np.maximum(r,np.maximum(g,b))
        Cmin = np.minimum(r,np.minimum(g,b))
        delta = Cmax - Cmin + 1e-7
        
        h[Cmax == r] = 60.0 * ((g[Cmax == r] - b[Cmax == r])/delta[Cmax == r] % 6.0)
        h[Cmax == g] = 60.0 * ((b[Cmax == g] - r[Cmax == g])/delta[Cmax == g] + 2.0)
        h[Cmax == b] = 60.0 * ((r[Cmax == b] - g[Cmax == b])/delta[Cmax == b] + 4.0)
        
        s[delta!=0.0] = delta[delta!=0.0] / (Cmax[delta!=0.0] + 1e-7)
        
        v = Cmax

        # ustvari izhodno sliko        
        oImage = np.zeros_like( iImage )        
        if colIdx.index( True ) == 0:
            oImage[0,:,:] = h; oImage[1,:,:] = s;  oImage[2,:,:] = v;
        elif colIdx.index( True ) == 1:
            oImage[:,0,:] = h; oImage[:,1,:] = s; oImage[:,2,:] = v;
        elif colIdx.index( True ) == 2:
            oImage[:,:,0] = h; oImage[:,:,1] = s; oImage[:,:,2] = v;
            
        return oImage
        
    elif iConversionType == 'HSVtoRGB':
        if colIdx.index( True ) == 0:
            h = iImage[0,:,:]; s = iImage[1,:,:];  v = iImage[2,:,:];
        elif colIdx.index( True ) == 1:
            h = iImage[:,0,:]; s = iImage[:,1,:]; v = iImage[:,2,:];
        elif colIdx.index( True ) == 2:
            h = iImage[:,:,0]; s = iImage[:,:,1]; v = iImage[:,:,2];    

        C = v * s
        X = C * (1.0 - np.abs( ( (h/60.0) % 2.0 ) - 1 ) )
        m = v - C

        r = np.zeros_like( h )
        g = np.zeros_like( h )
        b = np.zeros_like( h )        
        
        r[ (h>=0.0) * (h<60.0) ] = C[ (h>=0.0) * (h<60.0) ]
        g[ (h>=0.0) * (h<60.0) ] = X[ (h>=0.0) * (h<60.0) ]

        r[ (h>=60.0) * (h<120.0) ] = X[ (h>=60.0) * (h<120.0) ]
        g[ (h>=60.0) * (h<120.0) ] = C[ (h>=60.0) * (h<120.0) ]

        g[ (h>=120.0) * (h<180.0) ] = C[ (h>=120.0) * (h<180.0) ]
        b[ (h>=120.0) * (h<180.0) ] = X[ (h>=120.0) * (h<180.0) ]

        g[ (h>=180.0) * (h<240.0) ] = X[ (h>=180.0) * (h<240.0) ]
        b[ (h>=180.0) * (h<240.0) ] = C[ (h>=180.0) * (h<240.0) ]

        r[ (h>=240.0) * (h<300.0) ] = X[ (h>=240.0) * (h<300.0) ]
        b[ (h>=240.0) * (h<300.0) ] = C[ (h>=240.0) * (h<300.0) ]
        
        r[ (h>=300.0) * (h<360.0) ] = C[ (h>=300.0) * (h<360.0) ]
        b[ (h>=300.0) * (h<360.0) ] = X[ (h>=300.0) * (h<360.0) ]        
            
        r = r + m
        g = g + m
        b = b + m
        
        # ustvari izhodno sliko        
        oImage = np.zeros_like( iImage )
        print(oImage.dtype)
        if colIdx.index( True ) == 0:
            oImage[0,:,:] = r; oImage[1,:,:] = g;  oImage[2,:,:] = b;
        elif colIdx.index( True ) == 1:
            oImage[:,0,:] = r; oImage[:,1,:] = g; oImage[:,2,:] = b;
        elif colIdx.index( True ) == 2:
            oImage[:,:,0] = r; oImage[:,:,1] = g; oImage[:,:,2] = b;
        
        # zaokrozevanje vrednosti
        oImage = 255.0 * oImage
        oImage[oImage>255.0] = 255.0
        oImage[oImage<0.0] = 0.0
        
        oImage = np.array( oImage, dtype='uint8' )
        
        return oImage


def discreteConvolution2D( iImage, iKernel ):
    '''
    Diskretna 2D konvolucija slike s poljubnim jedrom

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iKernel : numpy.ndarray
        Jedro ali matrika za konvolucijo
        
    Returns
    --------
    oImage : numpy.ndarray
        Z jedrom konvolirana vhodna slika
    '''
    # pretvori vhodne spremenljivke v np polje in
    # inicializiraj izhodno np polje
    iImage = np.asarray( iImage )
    iKernel = np.asarray( iKernel )
    return convolve( iImage, iKernel, mode='nearest' )
    # direktna implementacija
    oImage = np.zeros_like( iImage ).astype('float')
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    dv, du = iKernel.shape
    # izracunaj konvolucijo
    for y in range( dy ):
        for x in range( dx ):
            for v in range( dv ):
                for u in range( du ):
                    tx = x - u + du/2
                    ty = y - v + dv/2
                    if tx>=0 and tx<dx and ty>=0 and ty<dy:
                        oImage[y, x] = oImage[y, x] + \
                            float(iImage[ty, tx]) * float(iKernel[v, u])
    if iImage.dtype.kind in ('u','i'):
        oImage[oImage<np.iinfo(iImage.dtype).min] = np.iinfo(iImage.dtype).min
        oImage[oImage>np.iinfo(iImage.dtype).max] = np.iinfo(iImage.dtype).max
    return np.array( oImage, dtype=iImage.dtype )


def discreteGaussian2D( iSigma ):
    '''
    Diskretno 2D Gaussovo jedro za glajenje slik

    Parameters
    ----------
    iSigma : float
        Standardna deviacija simetričnega 2D Gaussovega jedra
        
    Returns
    --------
    oKernel : numpy.ndarray
        2D Gaussovo jedro
    '''    
    iKernelSize = int(2 * np.ceil( 3 * iSigma ) + 1)
    oKernel = np.zeros([iKernelSize, iKernelSize])
    k2 = np.floor(iKernelSize/2); s2 = iSigma**2.0
    for y in range(oKernel.shape[1]):
        for x in range(oKernel.shape[0]):
            oKernel[y, x] = np.exp(-((x-k2)**2+(y-k2)**2)/2.0/s2 ) / s2 / 2.0 / np.pi
    return oKernel


def interpolate0Image2D( iImage, iCoorX, iCoorY ):
    '''
    Funkcija za interpolacijo ničtega reda

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iCoorX : numpy.ndarray
        Polje X koordinat za interpolacijo

    iCoorY : numpy.ndarray
        Polje Y koordinat za interpolacijo
        
    Returns
    --------
    oImage : numpy.ndarray
        Interpolirane vrednosti v vhodnih koordinatah X in Y
    '''
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray( iImage )    
    iCoorX = np.asarray( iCoorX )
    iCoorY = np.asarray( iCoorY )   
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    # ustvari 2d polje koordinat iz 1d vhodnih koordinat (!!!)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')      
        iCoorX, iCoorY = np.meshgrid(iCoorX, iCoorY, sparse=False, indexing='xy')
    return interpn((np.arange(dy),np.arange(dx)),iImage.astype('float'),
        np.dstack((iCoorY,iCoorX)),method='nearest',bounds_error=False).astype(iImage.dtype)        
    # zaokrozi na najblizjo celostevilsko vrednost (predstavlja indeks!)
    oShape = iCoorX.shape    
    iCoorX = np.round(iCoorX); iCoorX = iCoorX.flatten()
    iCoorY = np.round(iCoorY); iCoorY = iCoorY.flatten()
    # ustvari izhodno polje    
    oImage = np.zeros( oShape ); oImage = oImage.flatten()
    oImage = np.array( oImage, dtype=iImage.dtype )
    print(iCoorX.shape)
    print(iCoorY.shape)
    # priredi vrednosti    
    for idx in range(oImage.size):
        tx = iCoorX[idx]
        ty = iCoorY[idx]
        if tx>=0 and tx<dx and ty>=0 and ty<dy:
            oImage[idx] = iImage[ty, tx]
    # vrni izhodno sliko
    return np.reshape( oImage, oShape )


def interpolate1Image2D( iImage, iCoorX, iCoorY ):
    '''
    Funkcija za interpolacijo prvega reda

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iCoorX : numpy.ndarray
        Polje X koordinat za interpolacijo

    iCoorY : numpy.ndarray
        Polje Y koordinat za interpolacijo
        
    Returns
    --------
    oImage : numpy.ndarray
        Interpolirane vrednosti v vhodnih koordinatah X in Y
    '''
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray( iImage )    
    iCoorX = np.asarray( iCoorX )
    iCoorY = np.asarray( iCoorY )   
    # preberi velikost slike in jedra
    dy, dx = iImage.shape
    # ustvari 2d polje koordinat iz 1d vhodnih koordinat (!!!)
    if np.size(iCoorX) != np.size(iCoorY):
        print('Stevilo X in Y koordinat je razlicno!')      
        iCoorX, iCoorY = np.meshgrid(iCoorX, iCoorY, sparse=False, indexing='xy')
    return interpn((np.arange(dy),np.arange(dx)),iImage.astype('float'),
        np.dstack((iCoorY,iCoorX)),method='linear',bounds_error=False).astype(iImage.dtype)
    # pretvori v linearno polje
    oShape = iCoorX.shape    
    iCoorX = iCoorX.flatten()
    iCoorY = iCoorY.flatten()
    # ustvari izhodno polje, pretvori v linearno polje
    oImage = np.zeros( oShape ); oImage = oImage.flatten()
    oImage = np.array( oImage, dtype='float' )
    print(iCoorX.shape)
    print(iCoorY.shape)
    # priredi vrednosti    
    for idx in range(oImage.size):
        lx = np.floor(iCoorX[idx])
        ly = np.floor(iCoorY[idx])
        sx = float(iCoorX[idx]) - lx
        sy = float(iCoorY[idx]) - ly        
        if lx>=0 and lx<(dx-1) and ly>=0 and ly<(dy-1):
            # izracunaj utezi
            a = (1 - sx) * (1 - sy)
            b = sx * (1 - sy)
            c = (1 - sx) * sy
            d = sx * sy
            # izracunaj izhodno vrednost
            oImage[idx] = a * iImage[ly, lx] + \
                          b * iImage[ly, lx+1] + \
                          c * iImage[ly+1, lx] + \
                          d * iImage[ly+1, lx+1]
    if iImage.dtype.kind in ('u','i'):
        oImage[oImage<np.iinfo(iImage.dtype).min] = np.iinfo(iImage.dtype).min
        oImage[oImage>np.iinfo(iImage.dtype).max] = np.iinfo(iImage.dtype).max
    return np.array( np.reshape( oImage, oShape ), dtype=iImage.dtype ) 


def decimateImage2D( iImage, iLevel ):
    '''
    Funkcija za piramidno decimacijo

    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna sivinska slika

    iLevel : int
        Število decimacij s faktorjem 2
        
    Returns
    --------
    oImage : numpy.ndarray
        Decimirana slika
    '''
    print('Decimacija pri iLevel = ', iLevel)
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray( iImage )
    iImageType = iImage.dtype
    # gaussovo jedro za glajenje
    iKernel = np.array( ((1/16,1/8,1/16),(1/8,1/4,1/8),(1/16,1/8,1/16)) )
    # glajenje slike pred decimacijo
    # iImage = discreteConvolution2D( iImage, iKernel )
    # hitrejsa verzija glajenja
    iImage = convolve( iImage, iKernel, mode='nearest' )
    # decimacija s faktorjem 2
    iImage = iImage[::2,::2]
    # vrni sliko oz. nadaljuj po piramidi
    if iLevel <= 1:
        return np.array( iImage, dtype=iImageType )
    else:
        return decimateImage2D( iImage, iLevel-1 )

def transAffine2D(iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
    '''
    Ustvari poljubno 2D afino preslikavo v obliki 3x3 homogene matrike

    Parameters
    ----------
    iScale : tuple, list
        Skaliranje vzdolž x in y

    iTrans : tuple, list
        Translacija vzdolž x in y

    iRot : float
        Kot rotacije

    iShear : tuple, list
        Strig vzdolž x in y
        
    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika

    '''
    iRot = iRot * np.pi / 180
    oMatScale = np.array( ((iScale[0],0,0),(0,iScale[1],0),(0,0,1)) )
    oMatTrans = np.array( ((1,0,iTrans[0]),(0,1,iTrans[1]),(0,0,1)) )
    oMatRot = np.array(((np.cos(iRot),-np.sin(iRot),0),\
                        (np.sin(iRot),np.cos(iRot),0),
                        (0,0,1)))
    oMatShear = np.array( ((1,iShear[0],0),(iShear[1],1,0),(0,0,1)) )
    # ustvari izhodno matriko
    oMat2D = np.dot(oMatTrans, np.dot(oMatShear, np.dot(oMatRot, oMatScale)))
    return oMat2D
    
def addHomogCoord2D(iPts):
    '''
    Seznamu 2D koordinat dodaj homogeno koordinato

    Parameters
    ----------
    iPts : numpy.ndarray
        Polje Nx2 koordinat x in y
        
    Returns
    --------
    oPts : numpy.ndarray
        Polje Nx3 homogenih koordinat x in y

    '''
    iPts = np.asarray( iPts )
    if iPts.shape[-1] == 3:
        return iPts
    iPts = np.hstack((iPts, np.ones((iPts.shape[0],1))))
    return iPts
    
def addHomo( iPts ):
    iPts = np.asarray( iPts )
    iPts = np.hstack( (iPts, np.ones((iPts.shape[0],1))) )
    return iPts

def mapAffineInterp2D(iPtsRef, iPtsMov):
    '''
    Afina interpolacijska poravnava na osnovi 3 pripadajočih parov točk

    Parameters
    ----------
    iPtsRef : numpy.ndarray
        Polje 3x2 koordinat x in y (lahko tudi v homogeni obliki)
        
    iPtsMov : numpy.ndarray
        Polje 3x2 koordinat x in y (lahko tudi v homogeni obliki)

    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika
    '''
    # dodaj homogeno koordinato
    iPtsRef = addHomogCoord2D(iPtsRef)
    iPtsMov = addHomogCoord2D(iPtsMov)
    # afina interpolacija
    iPtsRef = iPtsRef.transpose()
    iPtsMov = iPtsMov.transpose()
    oMat2D =  np.dot(iPtsRef, np.linalg.inv(iPtsMov))
    return oMat2D


def mapAffineApprox2D(iPtsRef, iPtsMov, iUsePseudoInv=False):
    '''
    Afina aproksimacijska poravnava na osnovi N pripadajočih parov točk

    Parameters
    ----------
    iPtsRef : numpy.ndarray
        Polje Nx2 koordinat x in y (lahko tudi v homogeni obliki)
        
    iPtsMov : numpy.ndarray
        Polje Nx2 koordinat x in y (lahko tudi v homogeni obliki)

    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika
    '''
    if iUsePseudoInv:      
        # po potrebi dodaj homogeno koordinato
        iPtsRef = addHomogCoord2D(iPtsRef)
        iPtsMov = addHomogCoord2D(iPtsMov)
        # afina aproksimacija (s psevdoinverzom)
        iPtsRef = iPtsRef.transpose()
        iPtsMov = iPtsMov.transpose()            
        # psevdoinverz
        oMat2D = np.dot(iPtsRef, np.linalg.pinv(iPtsMov))        
        # psevdoinverz na dolgo in siroko:
        # oMat2D = iPtsMov * iPtsRef.transpose() * \
        # np.linalg.inv( iPtsRef * iPtsRef.transpose() )        
    else:
        # izloci koordinate            
        x = np.array(iPtsMov[:,0])
        y = np.array(iPtsMov[:,1])
        
        u = np.array(iPtsRef[:,0])
        v = np.array(iPtsRef[:,1])
        
        # doloci povprecja
        uxm = np.mean(u*x)
        uym = np.mean(u*y)
        vxm = np.mean(v*x)
        vym = np.mean(v*y)
        um = np.mean(u)
        vm = np.mean(v)
        xxm = np.mean(x*x)
        xym = np.mean(x*y)
        yym = np.mean(y*y)
        xm = np.mean(x) 
        ym = np.mean(y)
        # sestavi vektor in matriko linearnega sistema        
        pv = np.array((uxm,uym,um,vxm,vym,vm))
        Pm = np.array(((xxm,xym, xm,  0,  0,  0), \
                       (xym,yym, ym,  0,  0,  0), \
                       (xm , ym,  1,  0,  0,  0), \
                       (  0,  0,  0,xxm,xym, xm), \
                       (  0,  0,  0,xym,yym, ym), \
                       (  0,  0,  0, xm, ym,  1)))
        t = np.dot(np.linalg.inv(Pm), pv)
        oMat2D = np.array(((t[0], t[1], t[2]), \
                           (t[3], t[4], t[5]), \
                           (   0,    0,    1)))           
    return oMat2D

    
def findCorrespondingPoints(iPtsRef, iPtsMov):
    '''
    Iskanje pripadajočih parov točk kot paroma najbližje tocke

    Parameters
    ----------
    iPtsRef : numpy.ndarray
        Polje Mx2 koordinat x in y (lahko tudi v homogeni obliki)
        
    iPtsMov : numpy.ndarray
        Polje Nx2 koordinat x in y (lahko tudi v homogeni obliki)

    Returns
    --------
    oPtsRef : numpy.ndarray
        Polje Kx3 homogenih koordinat x in y (ali v homogeni obliki), ki pripadajo oPtsMov (K=min(M,N))

    oPtsMov : numpy.ndarray
        Polje Kx3 homogenih koordinat x in y (ali v homogeni obliki), ki pripadajo oPtsRef (K=min(M,N))
    '''
    # inicializiraj polje indeksov
    idxPair = -np.ones((iPtsRef.shape[0], 1)).astype('int32')
    idxDist = np.ones((iPtsRef.shape[0], iPtsMov.shape[0]))
    for i in range(iPtsRef.shape[0]):
        for j in range(iPtsMov.shape[0]):
            idxDist[i,j] = np.sum((iPtsRef[i,:2] - iPtsMov[j,:2])**2)
    # doloci bijektivno preslikavo
    while not np.all(idxDist==np.inf):            
        i, j = np.where(idxDist == np.min(idxDist))
        idxPair[i[0]] = j[0]
        idxDist[i[0],:] = np.inf
        idxDist[:,j[0]] = np.inf            
    # doloci pare tock
    idxValid, idxNotValid = np.where(idxPair>=0)
    idxValid = np.array( idxValid )               
    iPtsRef_t = iPtsRef[idxValid,:]
    iPtsMov_t = iPtsMov[idxPair[idxValid].flatten(),:]   
#        iPtsMov_t = np.squeeze(iPtsMov[idxPair[idxValid],:])  
    return iPtsRef_t, iPtsMov_t


def alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50):
    '''
    Postopek iterativno najblizje tocke

    Parameters
    ----------
    iPtsRef : numpy.ndarray
        Polje Mx2 koordinat x in y (lahko tudi v homogeni obliki)
        
    iPtsMov : numpy.ndarray
        Polje Nx2 koordinat x in y (lahko tudi v homogeni obliki)

    iEps : float
        Največja absolutna razlika do homogene matrike preslikave identitete, ki zaustavi postopek

    iMaxIter : int
        Maksimalno število iteracij

    Returns
    --------
    oMat2D : numpy.ndarray
        Homogena 3x3 transformacijska matrika med setoma vhodnih točk

    oErr : list
        Srednja Evklidska razdalja med pripadajočimi pari točk preko iteracij
    '''   
    # inicializiraj izhodne parametre
    curMat = []; oErr = []; iCurIter = 0
    # zacni iterativni postopek
    while True:
        # poisci korespondencne pare tock
        iPtsRef_t, iPtsMov_t = findCorrespondingPoints(iPtsRef, iPtsMov)
        # doloci afino aproksimacijsko preslikavo
        oMat2D = mapAffineApprox2D(iPtsRef_t, iPtsMov_t)
        # posodobi premicne tocke
        iPtsMov = np.dot(addHomogCoord2D(iPtsMov), oMat2D.transpose())
        # izracunaj napako
        curMat.append(oMat2D)
        oErr.append(np.sqrt(np.sum((iPtsRef_t[:,:2]- iPtsMov_t[:,:2])**2)))
        iCurIter = iCurIter + 1 
        # preveri kontrolne parametre        
        dMat = np.abs(oMat2D - transAffine2D())
        if iCurIter>iMaxIter or np.all(dMat<iEps):
            break
    # doloci kompozitum preslikav
    oMat2D = transAffine2D()
    for i in range(len(curMat)):
        oMat2D = np.dot(curMat[i], oMat2D)
    return oMat2D, oErr