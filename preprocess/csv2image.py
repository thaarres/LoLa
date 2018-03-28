#!/usr/bin/env python

import glob
import numpy as np
import pandas
import sys
import h5py
import math
import scipy
import scipy.ndimage.filters

#converts input four-vectors (from csv file) into jet image, a la DeepTop
#writes out in compressed hdf5 format

#DON'T USE YET -- possible bug in rotation...
#and slow as hell, skip pandas step and write direct numpy->hdf5??

# calculate pseudorapidity of pixel entries
def eta(pT, pz):
    assert len(pT) == len(pz)
    small = 1e-10
    etas = np.zeros(len(pz))
    for ix, x in enumerate(pz):

        #make infinite rapidity finite but large
        if abs(pT[ix]) < small:
            etas[ix] == 1e10
            continue
        if abs(pz[ix]) < small:
            etas[ix] == 0.
            continue        
        theta = math.atan(pT[ix]/pz[ix])
        if theta < 0: theta += math.pi
        etas[ix] -= math.log(math.tan(theta/2))
    return etas

#calculate phi (in range [-pi,pi]) of pixel entries
def phi(px, py):
    assert len(px)==len(py)
    small = 1e-10
    phis = np.zeros(len(px))
    for ix, x in enumerate(pz):
        if (abs(px[ix]) < small and abs(py[ix]) < small):
            phis[ix] = 0
            continue
        phis[ix] += math.atan2(py[ix],px[ix]) #between -pi and pi
#    if phis[ix] < 0: phis[ix] += math.pi #between 0 and 2pi
#    if phis[ix] > 2*math.pi: phis[ix] -= math.pi #between 0 and 2pi
    return phis

xpixels = np.arange(-3.6, 3.6, 0.1)
ypixels = np.arange(-180, 180, 5)
assert len(xpixels) == len(ypixels) #need a square grid
xgrid, ygrid = np.meshgrid(xpixels, ypixels)

#put eta-phi entries on grid
def orig_image(eta,phi,e):
    assert (len(eta) == len(phi) and len(eta) == len(e))
    z = np.zeros((len(xpixels),len(ypixels)))
    for ix, x in enumerate(e):

        #make sure pixel entry is in grid
        if eta[ix] < xpixels[0] or eta[ix] > xpixels[-1] or phi[ix] < ypixels[0] or phi[ix] > ypixels[-1]: continue
#        print ix, x
        xpixel = np.where( eta[ix] >= xpixels)[0][-1]
        ypixel = np.where( phi[ix] >= xpixels)[0][-1]
        z[xpixel,ypixel] += e[ix] #
    return z

#image preprocessing options
Shift, Rotate, Flip, Crop = False, False, False, False #return original image
Shift, Rotate, Flip, Crop = True, False, False, True #'minimal' preprocessing
#Shift, Rotate, Flip, Crop = True, True, True, True #the whole shebang

def preprocess_image(z,entry):

    Npix = z.shape #initial image size
    imgpix = int(40) if Crop else len(xpixels)

    #assume unsorted array of pixel intenstities
    xmax, ymax = np.unravel_index(np.argsort(z, axis=None), z.shape)

    p1 = [xmax[-1], ymax[-1]] #first maximum (at back of array)
    p2 = [xmax[-2], ymax[-2]] #second maximum
    p3 = [xmax[-3], ymax[-3]] #third maximum

    #shift maxima by 1/2-pixel, to get rotation right - @TODO: problem here? 
    center = np.matrix([ (xgrid[p1[0],p1[1]]+(xpixels[1]-xpixels[0])/2.)/xpixels.max(),
        (ygrid[p1[0],p1[1]]+(ypixels[1]-ypixels[0])/2.)/180.])
    second = np.matrix([ (xgrid[p2[0],p2[1]]+(xpixels[1]-xpixels[0])/2.)/xpixels.max(),
        (ygrid[p2[0],p2[1]]+(ypixels[1]-ypixels[0])/2.)/180.])
    third = np.matrix([ (xgrid[p3[0],p3[1]]+(xpixels[1]-xpixels[0])/2.)/xpixels.max(),
        (ygrid[p3[0],p3[1]]+(ypixels[1]-ypixels[0])/2.)/180.])

    #shift 1st maximum to origin
    if Shift:
        z = np.vstack((z,z,z))
        shift = [int(Npix[0]/2)-p1[0]-1, int(Npix[1]/2)-p1[1]-1]
        z_new = scipy.ndimage.interpolation.shift(z, shift, order=0)
        z_new=z_new[z_new.shape[0]/3:2*z_new.shape[0]/3,:]
        if p2[0]-p1[0] < -int(Npix[0]/2): # take care of periodicity in phi
            second[0,1]+=2.
        if p2[0]-p1[0] > int(Npix[0]/2):
            second[0,1]-=2.
        if p3[0]-p1[0] < -int(Npix[0]/2):
            third[0,1]+=2.
        if p3[0]-p1[0] > int(Npix[0]/2):
            third[0,1]-=2.
    else:
        z_new = z

    # rotate so that 2nd maximum is at 6 o'clock
    if Rotate:
        ex = np.matrix([[0,-1]])
        theta = np.arccos(( (-center+second)*ex.T/np.linalg.norm(center-second)))[0,0]
        if p2[1]<p1[1]:
            theta*=-1.
        z_new = scipy.ndimage.interpolation.rotate(z_new, theta*180./np.pi,reshape=False, cval=0,order=0)

    #flip so that 3rd maximum is always in right of image (x>0)
    if Flip:
        thirdp=(np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta),  np.cos(theta)]]).dot((third-center).T)).T
        if thirdp[0,0] < -0.001:
            z_new=np.fliplr(z_new)

    #finally, crop image to new pixel grid
    if Crop:
        Npix = z_new.shape
        z_new = z_new[Npix[0]/2-imgpix/2:Npix[0]/2+imgpix/2,Npix[1]/2-imgpix/2:Npix[1]/2+imgpix/2]

    if Plot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=[8,6.5])
        ax1 = fig.add_subplot(1,1,1)
        p1 = ax1.imshow(z_new, origin='lower', aspect='auto',interpolation='nearest',norm=mpl.colors.LogNorm()) 
        ax1.set_xlabel('$\\eta$')
        ax1.set_ylabel('$\\phi$')
        cb=plt.colorbar(p1)
        fig.savefig('test.png')

    return z_new
            
n_cands = 10
version = "v17_{0}nc".format(n_cands)

sigfile, bkgfile = 'sig.csv', 'bkg.csv'
df_sig = pandas.read_csv(sigfile,header=0)
df_bkg = pandas.read_csv(bkgfile,header=0)

df_sig["is_signal"] = 1
df_bkg["is_signal"] = 0
df = pandas.concat([df_sig, df_bkg], ignore_index=True)

#drop nans - @TODO: a more elegant way of doing this?
df.dropna(how='any',inplace=True)

#iterate through dataframe, build image for each row(jet)
for index, row in df.iterrows():
#    print index
    px = row.filter(like='PX').values
    py = row.filter(like='PY').values
    pz = row.filter(like='PZ').values
    e = row.filter(like='E').values

    pT = np.sqrt(px**2+py**2)#.sort()[::-1]
    eta_cand =  eta(pT,pz)
    phi_cand =  phi(px,py)

    z_orig = orig_image(eta_cand,phi_cand,e)
    z_preproc = preprocess_image(z_orig,index)

    #slow
#    for i in range(n_cands):            
#        df["eta_{0}".format(i)] = eta_cand[i]
#        df["phi_{0}".format(i)] = phi_cand[i]
    z_preproc = z_preproc.flatten()
    #@TODO: slow as hell, keep nonzero only?
    for i in range(len(z_preproc)):
        df["img_e_{0}".format(i)] = z_preproc[i]

# shuffle                
df = df.iloc[np.random.permutation(len(df))]

# Train / Test / Validate
# ttv==0: 60% Train
# ttv==1: 20% Test
# ttv==2: 20% Final Validation
df["ttv"] = np.random.choice([0,1,2], df.shape[0],p=[0.6, 0.2, 0.2])

train = df[ df["ttv"]==0 ]
test  = df[ df["ttv"]==1 ]
val   = df[ df["ttv"]==2 ]
        

train.to_hdf('top+qcdconst-train-{0}.h5'.format(version),'table',append=True)
test.to_hdf('top+qcdconst-test-{0}.h5'.format(version),'table',append=True)
val.to_hdf('top+qcdconst-val-{0}.h5'.format(version),'table',append=True)
