# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:16:43 2015

@author: Ziga Spiclin

Vaja 10: Sledenje in analiza gibanja
"""

import scipy.ndimage as ni
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sp
from scipy.interpolate import interpn
import matplotlib.animation as animation

#------------------------------------------------------------------------------
# POMOZNE FUNKCIJE
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
    plt.axes().set_aspect('equal', 'datalim')
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
    #plt.axes('equal') # should, but doesnt work
    plt.show()
    
def colorToGray( iImage ):
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
       
def discreteConvolution2D( iImage, iKernel ):
    """Diskretna 2D konvolucija slike s poljubnim jedrom"""    
    # pretvori vhodne spremenljivke v np polje in
    # inicializiraj izhodno np polje
    iImage = np.asarray( iImage )
    iKernel = np.asarray( iKernel )
    #------------------------------- za hitrost delovanja
    oImage = ni.convolve( iImage, iKernel, mode='nearest' )    
    return oImage

def imageGradient( iImage ):
    """Gradient slike s Sobelovim operatorjem"""
    iImage = np.array( iImage, dtype='float' )    
    iSobel = np.array( ((-1,0,1),(-2,0,2),(-1,0,1)) )    
    oGx = ni.convolve( iImage, iSobel, mode='nearest' )
    oGy = ni.convolve( iImage, np.transpose( iSobel ), mode='nearest' )
    return oGx, oGy

def decimateImage2D( iImage, iLevel ):
    """Funkcija za piramidno decimacijo"""  
#    print('Decimacija pri iLevel = ', iLevel)
    # pretvori vhodne spremenljivke v np polje
    iImage = np.asarray( iImage )
    iImageType = iImage.dtype
    # gaussovo jedro za glajenje
    iKernel = np.array( ((1/16,1/8,1/16),(1/8,1/4,1/8),(1/16,1/8,1/16)) )
    # glajenje slike pred decimacijo
    iImage = discreteConvolution2D( iImage, iKernel )
    # decimacija s faktorjem 2
    iImage = iImage[::2,::2]
    # vrni sliko oz. nadaljuj po piramidi
    if iLevel <= 1:
        return np.array( iImage, dtype=iImageType )
    else:
        return decimateImage2D( iImage, iLevel-1 )       
       
def interpolate1Image2D( iImage, iCoorX, iCoorY ):
    """Funkcija za interpolacijo prvega reda"""
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
    #------------------------------- za hitrost delovanja    
    return interpn( (np.arange(dy),np.arange(dx)), iImage, \
                      np.dstack((iCoorY,iCoorX)),\
                      method='linear', bounds_error=False)\
                      .astype( iImage.dtype )    
               
def transAffine2D( iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0) ):
    """Funkcija za poljubno 2D afino preslikavo"""    
    iRot = iRot * np.pi / 180
    oMatScale = np.matrix( ((iScale[0],0,0),(0,iScale[1],0),(0,0,1)) )
    oMatTrans = np.matrix( ((1,0,iTrans[0]),(0,1,iTrans[1]),(0,0,1)) )
    oMatRot = np.matrix( ((np.cos(iRot),-np.sin(iRot),0),\
                          (np.sin(iRot),np.cos(iRot),0),(0,0,1)) )
    oMatShear = np.matrix( ((1,iShear[0],0),(iShear[1],1,0),(0,0,1)) )
    # ustvari izhodno matriko
    oMat2D = oMatTrans * oMatShear * oMatRot * oMatScale
    return oMat2D               
               
def transformImage( iImage, oMat2D ):
    """Preslikaj 2D sliko z linearno preslikavo"""
    # ustvari diskretno mrezo tock
    gx, gy = np.meshgrid( range(iImage.shape[1]), \
                          range(iImage.shape[0]), \
                          indexing = 'xy' )    
    # ustvari Nx3 matriko vzorcnih tock                          
    pts = np.vstack( (gx.flatten(), gy.flatten(), np.ones( (gx.size,))) ).transpose()
    # preslikaj vzorcne tocke
    pts = np.dot( pts, oMat2D.transpose() )
    # ustvari novo sliko z interpolacijo sivinskih vrednosti
    oImage = interpolate1Image2D( iImage, \
                                  pts[:,0].reshape( gx.shape ), \
                                  pts[:,1].reshape( gx.shape ) )
    oImage[np.isnan( oImage )] = 0
    return oImage 

def showVideo( oVideo, oPathXY=np.array([]) ):
    """Prikazi video animacijo poti"""
    global oVideo_t, iFrame, oPathXY_t
    fig = plt.figure()
    # prikazi prvi okvir
    iFrame = 0
    oPathXY_t = oPathXY
    oVideo_t = oVideo
    print(oVideo.shape)
    im = plt.imshow(oVideo[...,iFrame], cmap=plt.get_cmap('Greys_r'))
    # definiraj funkcijo za osvezevanje prikaza
    def updatefig(*args):
        global oVideo_t, iFrame, oPathXY_t
        iFrame = ( iFrame + 1 ) % oVideo_t.shape[-1]
        im.set_array( oVideo_t[...,iFrame] ) 
        if iFrame < oPathXY.shape[0]:
            plt.plot( oPathXY[iFrame,0], oPathXY[iFrame,1], 'xr' ,markersize=3 )    
        return im,
    # prikazi animacijo poti
    ani = animation.FuncAnimation(fig, updatefig, interval=25, blit=True)
    plt.show()  

def drawPathToFrame( oVideo, oPathXY, iFrame=1, iFrameSize=(40,40) ):    
    """Prikazi pot do izbranega okvirja"""
    oPathXY_t = oPathXY[:iFrame,:]
    showImage( oVideo[...,iFrame], 'Pot do okvirja %d' % iFrame )
    for i in range(1,oPathXY_t.shape[0]):
        plt.plot(oPathXY_t[i-1:i+1,0],oPathXY_t[i-1:i+1,1],'--r')
        if i==1 or (i%5)==0:
            plt.plot( oPathXY_t[i,0],oPathXY_t[i,1],'xr',markersize=3)
        
    dx = iFrameSize[0]/2; dy = iFrameSize[1]/2
    plt.plot( (oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]+dy,oPathXY_t[-1,1]+dy),'-g')   
    plt.plot( (oPathXY_t[-1,0]+dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]+dy),'-g')   
    plt.plot( (oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]-dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]+dy),'-g')
    plt.plot( (oPathXY_t[-1,0]-dx,oPathXY_t[-1,0]+dx),(oPathXY_t[-1,1]-dy,oPathXY_t[-1,1]-dy),'-g')
   
#%% Naloga 1: Nalozi video
# za nalaganje videa boste potrebovali knjiznico ffmpeg (datoteko ffmpeg.exe),
# ki jo lahko nalozite s spletne strani https://www.ffmpeg.org/download.html
def loadVideo( iFileName, iFrameSize = (576, 720) ):
    """Nalozi video z ffmpeg orodjem"""
    import sys
    import subprocess as sp
    # ustvari klic ffmpeg in preusmeri izhod v cevovod
    command = [ 'ffmpeg',
                '-i', iFileName,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
    # definiraj novo spremeljivko
    oVideo = np.array([])
    iFrameSize = np.asarray( iFrameSize )
    frameCount = 0
    # zacni neskoncno zanko
    while True:
        frameCount += 1
#        print( 'Berem okvir %d ...' % frameCount )
        print("\rBerem okvir %d ..." % frameCount, end="")
        # preberi Y*X*3 bajtov (= 1 okvir)
        raw_frame = pipe.stdout.read(np.prod(iFrameSize)*3)
        # pretvori prebrane podatke v numpy polje
        frame =  np.fromstring(raw_frame, dtype='uint8')       
        # preveri ce je velikost ustrezna, sicer prekini zanko
        if frame.size != (np.prod(iFrameSize)*3):
            print(" koncano!\n")
            break;
        # preoblikuj dimenzije in pretvori v sivinsko sliko
        frame = colorToGray( frame.reshape((iFrameSize[0],iFrameSize[1],3)) )
        # sprazni medpomnilnik        
        pipe.stdout.flush()    
        # vnesi okvir v izhodno sprememnljivko
        if oVideo.size == 0:
            oVideo = frame
            oVideo = oVideo[...,None]
        else:
            oVideo = np.concatenate((oVideo,frame[...,None]), axis=2)
    # zapri cevovod
    pipe.terminate()
    # vrni izhodno spremenljivko
    return oVideo
    
# test funkcije
if __name__ == '__main__':
    # nalozi video
    oVideo = loadVideo( 'video1.avi' )
    # prikazi prvi okvir
    plt.close('all')
    showImage( oVideo[..., 0] )
    # prikazi video
    plt.close('all')    
    showVideo( oVideo )

#%% Naloga 2: Funkcija za poravnavo z Lucas-Kanade postopkom
def regLucasKanade( iImgFix, iImgMov, iMaxIter, oPar = (0,0), iVerbose=True ):
    """Postopek poravnave Lucas-Kanade"""
    # pretvori vhodne slike v numpy polja tipa float
    iImgType = np.asarray( iImgMov ).dtype
    iImgFix = np.array( iImgFix, dtype='float' )
    iImgMov = np.array( iImgMov, dtype='float' )
    # doloci zacetne parametre preslikae
    oPar = np.array( oPar )     
    # izracunaj prva odvoda slike
    Gx, Gy = imageGradient( iImgMov )      
    # v zanki iterativno posodabljaj parametre
    for i in range( iMaxIter ):
        # doloci preslikavo pri trenutnih parametrih        
        oMat2D = transAffine2D( iTrans=oPar )        
        # preslikaj premicno sliko in sliki odvodov        
        iImgMov_t = transformImage( iImgMov, oMat2D )
        Gx_t = transformImage( Gx, oMat2D )
        Gy_t = transformImage( Gy, oMat2D )        
        # izracunaj sliko razlike in sistemsko matriko
        I_t = iImgMov_t - iImgFix
        B = np.vstack( (Gx_t.flatten(), Gy_t.flatten()) ).transpose()
        # resi sistem enacb
        invBtB = np.linalg.inv( np.dot( B.transpose(), B ) )
        dp = np.dot( np.dot( invBtB, B.transpose() ), I_t.flatten() )        
        # posodobi parametre        
        oPar = oPar + dp.flatten()           
        if iVerbose: print('iter: %d' % i, ', oPar: ', oPar)
    # doloci preslikavo pri koncnih parametrih        
    oMat2D = transAffine2D( iTrans=oPar )        
    # preslikaj premicno sliko        
    oImgReg = transformImage( iImgMov, oMat2D ).astype( iImgType )
    # vrni rezultat
    return oPar, oImgReg

# test funkcije
if __name__ == '__main__':
    # doloci fiksno in premicno sliko
    oPar = [0, 1]
    iImgFix = oVideo[:,:,0]
    iImgMov = transformImage( iImgFix, transAffine2D( iTrans = oPar ) )
    # klici Lucas-Kanade poravnavo slik
    import time    
    ts = time.clock()    
    oPar, oImgReg = regLucasKanade( iImgFix, iImgMov, 20 )
    print('parameters: ', oPar)
    print('elapsed time: ', 1000.0*(time.clock()-ts), ' ms')  
    # narisi rezultate
    plt.close('all')
    showImage( iImgFix.astype('float') - iImgMov.astype('float'), 'Pred poravnavo' )
    showImage( iImgFix.astype('float') - oImgReg.astype('float'), 'Po poravnavi' )

#%% Naloga 3: Funkcija za poravnavo s piramidnim Lucas-Kanade postopkom
def regPyramidLK( iImgFix, iImgMov, iMaxIter, iNumScales, iVerbose=True ):
    """Piramidna implementacija poravnave Lucas-Kanade"""
    # pretvori vhodne slike v numpy polja tipa float
    iImgFix = np.array( iImgFix, dtype='float' )
    iImgMov = np.array( iImgMov, dtype='float' )
    # pripravi piramido slik
    iPyramid = [ (iImgFix, iImgMov) ]
    for i in range(1,iNumScales):
        # decimiraj fiksno in premicno sliko za faktor 2
        iImgFix_2 = decimateImage2D( iImgFix, i )
        iImgMov_2 = decimateImage2D( iImgMov, i )
        # dodaj v seznam
        iPyramid.append( (iImgFix_2,iImgMov_2) )
    # doloci zacetne parametre preslikave
    oPar = np.array( (0,0) )          
    # izvedi poravnavo od najmanjse do najvecje locljivosti slik
    for i in range(len(iPyramid)-1,-1,-1):
        if iVerbose: print('PORAVNAVA Z DECIMACIJO x%d' % 2*i)
        # posodobi parametre preslikave
        oPar = oPar * 2.0
        # izvedi poravnavo pri trenutni locljivosti
        oPar, oImgReg = regLucasKanade( iPyramid[i][0], iPyramid[i][1], \
                            iMaxIter, oPar, iVerbose=iVerbose )
    # vrni koncne parametre in poravnano sliko
    return oPar, oImgReg

# test funkcije
if __name__ == '__main__':
    # doloci fiksno in premicno sliko
    oPar = [0, 10]
    iImgFix = oVideo[:,:,0]
    iImgMov = transformImage( iImgFix, transAffine2D( iTrans = oPar ) )
    # klici Lucas-Kanade poravnavo slik
    import time    
    ts = time.clock()    
    oPar, oImgReg = regPyramidLK( iImgFix, iImgMov, 20, 3 )
    print('parameters: ', oPar)
    print('elapsed time: ', 1000.0*(time.clock()-ts), ' ms')  
    # narisi rezultate
    plt.close('all')
    showImage( iImgFix.astype('float') - iImgMov.astype('float'), 'Pred poravnavo' )
    showImage( iImgFix.astype('float') - oImgReg.astype('float'), 'Po poravnavi' )
    
#%% Naloga 4: Funkcija za sledenje tarci z Lucas-Kanade postopkom
def trackTargetLK( iVideoMat, iCenterXY, iFrameXY, iVerbose=True ):
    """Postopek sledenja Lucas-Kanade"""
    # pretvori vhodni video v numpy polje
    iVideoMat = np.asarray( iVideoMat )
    iCenterXY = np.array( iCenterXY )
    # definiraj izhodno spremenljivko
    oPathXY = np.array( iCenterXY.flatten() ).reshape( (1,2) )
    # definiraj koordinate v tarci
    gx, gy = np.meshgrid( range(iFrameXY[0]), range(iFrameXY[1]) )
    gx = gx - float(iFrameXY[0]-1)/2.0
    gy = gy - float(iFrameXY[1]-1)/2.0
    # zazeni LK preko vseh zaporednih okvirjev
    for i in range(1,iVideoMat.shape[-1]):
#        print('PORAVNAVA OKVIRJEV %d-%d' % (i-1,i) )
        # vzorcni tarco v dveh zaporednih okvirjih        
        iImgFix = interpolate1Image2D( iVideoMat[...,i-1], \
                    gx+oPathXY[-1,0], gy+oPathXY[-1,1] )
        iImgMov = interpolate1Image2D( iVideoMat[...,i], \
                    gx+oPathXY[-1,0], gy+oPathXY[-1,1] )
        # zazeni piramidno LK poravnavo
        oPar, oImgReg = regPyramidLK( iImgFix, iImgMov, 30, 3, iVerbose=False )
        # shrani koordinate
        oPathXY = np.vstack( (oPathXY, oPathXY[-1,:] + oPar.flatten()) )     
        print('koordinate tarce: ', oPathXY[-1,:])
    # vrni spremenljivko
    return oPathXY

# test funkcije
if __name__ == '__main__':
    # klici Lucas-Kanade sledenje tarci
    import time    
    ts = time.clock()    
    oPathXY = trackTargetLK( oVideo[...,:], (33,370), (40,40) )
    print('elapsed time: ', 1000.0*(time.clock()-ts), ' ms')  
    # narisi rezultate
    plt.close('all')
    # prikazi tarco in pot v razlicnih okvirjih 
    drawPathToFrame( oVideo, oPathXY, iFrame=1, iFrameSize=(40,40),  )
    drawPathToFrame( oVideo, oPathXY, iFrame=100, iFrameSize=(40,40) )
    drawPathToFrame( oVideo, oPathXY, iFrame=170, iFrameSize=(40,40) )   
