import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def loadImageRaw(iPath, iSize, iFormat):
    raw_data = np.fromfile(iPath, dtype = iFormat) #surovi podatki
    oImage = np.reshape(raw_data, iSize) # dimenzije, ki so podane
    
    return oImage

def showImage(iImage, iTitle=''):
    #plt.figure(figsize = (12,12))
    plt.imshow(iImage, cmap='gray')
    plt.title(iTitle)
    plt.xlabel('x')
    plt.ylabel('y')
    
def convertToGray(image):
    dtype = image.dtype
    rgb = np.array(image, dtype='float')
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    
    return gray.astype(dtype)

def scaleImage(iImage, iSlopeA, iIntersectionB):
    iImageType = iImage.dtype
    iImage = iImage.astype('float')
    #iImage = np.array(iImage, dtype = 'float')
    oImage = iSlopeA * iImage + iIntersectionB
    if iImageType.kind in ('u', 'i'):
        #oImage[oImage < np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        max_val = np.iinfo(iImageType).max
        min_val = np.iinfo(iImageType).min
        oImage[oImage < min_val] = min_val
        oImage[oImage > max_val] = max_val
    return oImage.astype(iImageType)

def windowImage(iImage, iCenter, iWidth):
    iImageType = iImage.dtype
    if iImageType.kind in ('u', 'v'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:#ce niso cela stevila
        iMaxValue = np.max(iImage)
        iMinValue = np.min(iImage)
        iRange = iMaxValue - iMinValue
    
    iSlopeA = iRange / float(iWidth)
    iInterceptB = -iSlopeA * (float(iCenter) - iWidth/2.0)
    
    return scaleImage(iImage, iSlopeA, iInterceptB)

def thresholdImage(iImage, iThreshold):
    oImage = 255 * np.array(iImage > iThreshold, dtype='uint8')
    return oImage

def gammaImage(iImage, iGamma):
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype ='float')
    if iImageType.kind in ('u', 'v'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max(iImage)
        iMinValue = np.min(iImage)
        iRange = iMaxValue - iMinValue
    iImage = (iImage - iMinValue) / float(iRange)
    oImage = iImage ** iGamma
    oImage = float(iRange) * oImage + iMinValue
    
    if iImageType.kind in ('u', 'i'):
        #oImage[oImage < np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        max_val = np.iinfo(iImageType).max
        min_val = np.iinfo(iImageType).min
        oImage[oImage < min_val] = min_val
        oImage[oImage > max_val] = max_val
    return np.array(oImage, dtype=iImageType)
    