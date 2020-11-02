import PIL.Image as im
import scipy.ndimage as ni
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os.path
import sys

#------------------------------------------------------------------------------
# POMOZNE FUNKCIJE

def imageGradient( iImage ):
    """Gradient slike s Sobelovim operatorjem"""
    iImage = np.array( iImage, dtype='float' )    
    iSobel = np.array( ((-1,0,1),(-2,0,2),(-1,0,1)) )    
    oGx = ni.convolve( iImage, iSobel, mode='nearest' )
    oGy = ni.convolve( iImage, np.transpose( iSobel ), mode='nearest' )
    return oGx, oGy
                          

# LOAD IMAGE FROM FILE
def loadImage(iPath):
    
    img = im.open(iPath)
    arr_img = np.array(img)
    return arr_img                    
                      
def showImage( iImage, iTitle='', iTranspose=False, iCmap=cm.Greys_r ):
    """Prikazi sliko v lastnem naslovljenem prikaznem oknu"""
    # preslikaj koordinate barvne slike    
    if len(iImage.shape)==3 and iTranspose:
        iImage = np.transpose( iImage, [1,2,0])
    plt.figure()
    if iImage.dtype.kind in ('u','i'):
        vmin_ui = np.iinfo(iImage.dtype).min
        vmax_ui = np.iinfo(iImage.dtype).max
        plt.imshow(iImage, cmap = iCmap, vmin=vmin_ui, vmax=vmax_ui)
    else:
        plt.imshow(iImage, cmap = iCmap)
    plt.suptitle( iTitle )
    plt.xlabel('Koordinata x')
    plt.ylabel('Koordinata y')
    # podaj koordinate in indeks slike
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%s @ [%4i, %4i]" % (iImage[y, x], x, y)
        except IndexError:
            return "IndexError"    
    plt.gca().format_coord = format_coord
    
def toGray( iImage ):
    """Pretvori v sivinsko sliko"""
    iImage = np.asarray( iImage )
    iImageType = iImage.dtype
    colIdx = [iImage.shape[i] == 3 for i in range(len(iImage.shape))]
    
    if colIdx.index( True ) == 0:
        iImageG = 0.299 * iImage[0,:,:] + 0.587 * iImage[1,:,:] + 0.114 * iImage[2,:,:]
    elif colIdx.index( True ) == 1:
        iImageG = 0.299 * iImage[:,0,:] + 0.587 * iImage[:,1,:] + 0.114 * iImage[:,2,:]
    elif colIdx.index( True ) == 2:
        iImageG = 0.299 * iImage[:,:,0] + 0.587 * iImage[:,:,1] + 0.114 * iImage[:,:,2]
    
    return np.array( iImageG, dtype = iImageType )

# funkcija za pretvorbo sivinskih vrednosti v indekse
def getIndices( iData, iBins, iMinVal, iMaxVal ):      
    # pretvori v indeks polja
    idx = np.round( (iData - iMinVal) / (iMaxVal - iMinVal) * (iBins-1) )
    # vrni indekse
    return idx.astype('uint32')
    
# funkcija za izracun 1D histograma 
def hist1D( iData, iBins, iMinVal=None, iMaxVal=None ):
    # doloci obmocje sivinskih vrednosti  
    if iMinVal==None: iMinVal = np.min(iData)     
    if iMaxVal==None: iMaxVal = np.max(iData)+1e-7    
    # pretvorba sivinskih vrednosti v indekse
    idx = getIndices( iData, iBins, iMinVal, iMaxVal )
    # izracunaj histogram
    histData = np.zeros((iBins,))
    for i in idx:
        histData[i] += 1.0
    # vrni histogram
    return histData