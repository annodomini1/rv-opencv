# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:18:19 2015

@author: Ziga Spiclin

renderSurface()

Funkcija izlusci 3d trikotnisko mrezo iz danih tock in 
3d trikotnisko mrezo narise v novem prikaznem oknu.

iCoorX, iCoorY, iCoorZ: 
    x-, y-, z-koordinate tock
iTitle: 
    Naslov prikaznega okna
nSamples:
    Maksimalno stevilo tock v trikotniski mrezi

"""
import matplotlib.pyplot as plt
import numpy as np
    
def renderSurface( iCoorX, iCoorY, iCoorZ, iTitle='', nSamples=2000):
    """Izris 3D objekta s trikotnisko mrezo"""
    # Nalozi Python pakete
    from scipy.spatial import Delaunay
    import mpl_toolkits.mplot3d as a3
    import matplotlib.colors as colors
    import scipy as sp
    
    # UREDI TOCKE
    # pretvori koordinate v 1d polje
    iCoorX = iCoorX.flatten()
    iCoorY = iCoorY.flatten()
    iCoorZ = iCoorZ.flatten()
    # zdruzi tocke v vektor    
    pts = np.vstack((iCoorX,iCoorY,iCoorZ))
#    pts = pts.reshape((3,iCoorX.size))
    # zmanjsaj stevilo tock [nakljucno izberi cca. 2000 tock]
    idx = np.unique( np.floor( pts.shape[1] * sp.rand(nSamples) \
            ).astype('uint32') )
    pts = pts[:,idx]
    
    # PRETVORI TOCKE V SFERICNI KOORDINATNI SISTEM
    # izracunaj sredisce in centriraj
    ptsM = np.mean(pts,axis=1)
    ptsC = pts - np.tile( ptsM, (pts.shape[1],1)).transpose()
    # doloci radij od sredisca in sfericna kota [theta, phi]
    r = np.sqrt(ptsC[0,:]**2+ptsC[1,:]**2+ptsC[2,:]**2)
    sphpts = np.zeros( (2, ptsC.shape[1]) )
    sphpts[0,:] = np.arccos( ptsC[2,:] / r ) # theta
    sphpts[1,:] = np.arctan2( ptsC[1,:], ptsC[0,:] ) # phi
       
    # POVEZI TOCKE V MREZO Z DELAUNAY TRIANGULACIJO
    dl = Delaunay( sphpts.transpose() )

    # IZRISI MREZO S KVAZI BARVO KOZE    
    ax = a3.Axes3D(plt.figure())
    minAx = np.min(ptsC)
    maxAx = np.max(ptsC)
    ax.set_aspect('equal')
    ax.set_zlim(bottom=minAx,top=maxAx)
    ax.set_ylim(bottom=minAx,top=maxAx)
    ax.set_xlim(left=minAx, right=maxAx)
    skincol = colors.rgb2hex(np.array( (1,.75,.65) ))
    for i in range(0,dl.simplices.shape[0]):
        vtx = ptsC[:,dl.simplices[i,:]].transpose()
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(skincol)
        # tri.set_edgecolor(skincol)
        tri.set_edgecolor('k')
        tri.set_linewidth(.1)
        ax.add_collection3d(tri)    
    plt.suptitle( iTitle )
    plt.show()

# test funkcije
if __name__ == '__main__':
    glave = np.load('glave.npy')
    iGlava = 0 # 0 1 2 3 4 
    pts = glave[ iGlava ] 
    xs, ys, zs = np.transpose(pts)
    plt.close('all')
    renderSurface( xs, ys, zs, 'Glava ' + str(iGlava) )