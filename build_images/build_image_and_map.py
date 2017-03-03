import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ROOT
from ROOT import TFile, TLorentzVector, TTree
from array import array
import sys

if len(sys.argv) != 3:
    print "Provide infile and outfile"
    sys.exit()
    
#set size of final image
imgpix = int(40)

#image processing options
#Shift, Rotate, Flip, Chop = True, False, False, True #minimal
Shift, Rotate, Flip, Chop = True, True, True, True #full

# define grid: (x,y) <-> (eta, phi), 72x72, start from quadaratic grid. The setup should match Delphes output
x = np.arange(-3.6, 3.6, 0.1)
y = np.arange(-180, 180, 5)
xi, yi = np.meshgrid(x, y)


def mod_phi(phi): # [0, 2 pi] -> [-pi, pi]
    if phi > np.pi:
        phi = - 2. * np.pi + phi
    return phi

def raw_image(xvalues, yvalues, zvalues):
    z = np.zeros((len(y),len(x)), dtype=float)
    for i in xrange(len(xvalues)):
        ix = np.where(xvalues[i]>=x)[0][-1]
        iy = np.where(yvalues[i]*180./np.pi>=y)[0][-1]
        z[iy,ix] = zvalues[i]
    return z

def aligned_image(z, zz):
    z_orig = z
    zz_orig = zz
    Npix = z.shape
    
    # locate 3 leading maxima
    import scipy
    import scipy.ndimage.filters as filters
    maximap = (z == filters.maximum_filter(z,size=(3,3)))
    [maximapi, maximapj], maximapz = maximap.nonzero(), z[maximap]
    maskp = np.argsort(maximapz)[::-1]
    p1 = [maximapi[maskp][0],maximapj[maskp][0]]
    p2 = [maximapi[maskp][1],maximapj[maskp][1]]
    p3 = [maximapi[maskp][2],maximapj[maskp][2]]
    center = np.matrix([ (xi[p1[0],p1[1]]+(x[1]-x[0])/2.)/x.max(),
        (yi[p1[0],p1[1]]+(y[1]-y[0])/2.)/180.])
    second = np.matrix([ (xi[p2[0],p2[1]]+(x[1]-x[0])/2.)/x.max(),
        (yi[p2[0],p2[1]]+(y[1]-y[0])/2.)/180.])
    third = np.matrix([ (xi[p3[0],p3[1]]+(x[1]-x[0])/2.)/x.max(),
        (yi[p3[0],p3[1]]+(y[1]-y[0])/2.)/180.])

    
    # shift maximum to origin
    z, zz = np.vstack((z,z,z)), np.vstack((zz,zz,zz))
    shift = [int(Npix[0]/2)-p1[0]-1, int(Npix[1]/2)-p1[1]-1]
    zp = scipy.ndimage.interpolation.shift(z, shift, order=0)
    zzp = scipy.ndimage.interpolation.shift(zz, shift, order=0)
    zp=zp[zp.shape[0]/3:2*zp.shape[0]/3,:]
    zzp = zzp[zzp.shape[0]/3:2*zzp.shape[0]/3,:]
    if p2[0]-p1[0] < -int(Npix[0]/2): # take care of periodicity in phi
        second[0,1]+=2.
    if p2[0]-p1[0] > int(Npix[0]/2):
        second[0,1]-=2.
    if p3[0]-p1[0] < -int(Npix[0]/2):
       third[0,1]+=2.
    if p3[0]-p1[0] > int(Npix[0]/2):
        third[0,1]-=2.
   
    # rotate 2nd maximum to negativ y-axis
    ex = np.matrix([[0,-1]])
    theta = np.arccos(( (-center+second)*ex.T/np.linalg.norm(center-second)))[0,0]
    if p2[1]<p1[1]:
        theta*=-1.
    zpp = scipy.ndimage.interpolation.rotate(zp, theta*180./np.pi,reshape=False, cval=0,order=0)
    zzpp = scipy.ndimage.interpolation.rotate(zzp, theta*180./np.pi,reshape=False, cval=0,order=0)

    if not Rotate:
        zpp = zp
        zzpp = zzp  # shift image only, no rotation

    # flip such that 3rd maximum has positive x
    if Flip:
        thirdp=(np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta),  np.cos(theta)]]).dot((third-center).T)).T
        if thirdp[0,0] < -0.001:
            zpp=np.fliplr(zpp)
            zzpp = np.fliplr(zzpp)
        if zpp.any() < 0: print "alignment ", zpp

    if not Shift:    #no preprocessing, just return original image
        zpp = z_orig
        zzpp = zz_orig

    return zpp, zzpp

def aligned_image_jetaxis(z, jet):
    Npix = z.shape

    zjet = raw_image([jet.Eta()], [jet.Phi()], [100.])
    
    # locate maximum
    import scipy
    import scipy.ndimage.filters as filters
    maximap = (zjet == filters.maximum_filter(zjet,size=(3,3)))
    [maximapi, maximapj], maximapz = maximap.nonzero(), zjet[maximap]
    maskp = np.argsort(maximapz)[::-1]
    p1 = [maximapi[maskp][0],maximapj[maskp][0]]
    center = np.matrix([ (xi[p1[0],p1[1]]+(x[1]-x[0])/2.)/x.max(),
        (yi[p1[0],p1[1]]+(y[1]-y[0])/2.)/180.])  
    # shift maximum to origin
    z = np.vstack((z,z,z))
    shift = [int(Npix[0]/2)-p1[0]-1, int(Npix[1]/2)-p1[1]-1]
    zp = scipy.ndimage.interpolation.shift(z, shift, order=0)
    zp=zp[zp.shape[0]/3:2*zp.shape[0]/3,:]
    return zp

def chopped_image(z,imgpix):
    Npix = z.shape
    if z.any() < 0: print "chopped ",  z
    return z[Npix[0]/2-imgpix/2:Npix[0]/2+imgpix/2,Npix[1]/2-imgpix/2:Npix[1]/2+imgpix/2]

fin = TFile(sys.argv[1])
tin = fin.Get( 'tree' )

fout = ROOT.TFile(sys.argv[2], "recreate")
tout = ROOT.TTree("tree", "tree")
fatjet = ROOT.TLorentzVector(1.,2.,3.,4.)
filtered = ROOT.TLorentzVector(1.,2.,3.,4.)
top = ROOT.TLorentzVector(1.,2.,3.,4.)
softdropped = ROOT.TLorentzVector(1.,2.,3.,4.)
tau2 = np.zeros(1, dtype=float)
tau3 = np.zeros(1, dtype=float)
tau2_filt = np.zeros(1, dtype=float)
tau3_filt = np.zeros(1, dtype=float)
tau2_sd = np.zeros(1, dtype=float)
tau3_sd = np.zeros(1, dtype=float)
htt_tag = np.zeros(1, dtype=bool)
img_e = np.zeros(imgpix**2, dtype=float)
img_et = np.zeros(imgpix**2, dtype=float)
img_pt = np.zeros(imgpix**2, dtype=float)
img_min = np.zeros(imgpix**2, dtype=float)
entry = np.zeros(1, dtype=int)

# create the branches and assign the fill-variables to them
tout.Branch('fatjet', 'TLorentzVector', fatjet)
tout.Branch('filtered', 'TLorentzVector', filtered)
tout.Branch('softdropped', 'TLorentzVector', softdropped)
tout.Branch('top', 'TLorentzVector', top)
tout.Branch('tau2', tau2, 'tau2/D')
tout.Branch('tau3', tau3, 'tau3/D')
tout.Branch('tau2_filt', tau2_filt, 'tau2_filt/D')
tout.Branch('tau3_filt', tau3_filt, 'tau3_filt/D')
tout.Branch('tau2_sd', tau2_sd, 'tau2_sd/D')
tout.Branch('tau3_sd', tau3_sd, 'tau3_sd/D')
tout.Branch('htt_tagged', htt_tag, 'htt_tag/O')
tout.Branch('img_e', img_e, 'img['+str(imgpix**2)+']/D')
tout.Branch('img_et', img_et, 'img['+str(imgpix**2)+']/D')
tout.Branch('img_pt', img_pt, 'img['+str(imgpix**2)+']/D')
tout.Branch('img_min', img_min, 'img_min['+str(imgpix**2)+']/D')
tout.Branch('entry', entry, 'entry/I')

entries = tin.GetEntriesFast()
entries = 300000
n_negs=0

a_avg, b_avg, z_avg = np.zeros([72,72]), np.zeros([40,40]), np.zeros([72,72])

for jentry in xrange(entries):

    if jentry % 10000 == 0: print "At ", jentry 
    phi = []
    eta = []
    e, et, pt = [], [], []
    tin.GetEntry(jentry)
    if tin.veta.size() != tin.vet.size():
        print "Dodgy entry ", entry
        continue
    for k in range(tin.veta.size()):
        phi.append(mod_phi(tin.vphi[k]))
        eta.append(tin.veta[k])
        e.append(tin.ve[k])
        et.append(tin.vet[k])
        pt.append(tin.vpt[k])
    fat=tin.fatjet
    parton=tin.top 
    filt=tin.filtered
    sd=tin.softdropjet
    
    fatjet.SetPxPyPzE(fat.Px(),fat.Py(),fat.Pz(),fat.E())
    filtered.SetPxPyPzE(filt.Px(),filt.Py(),filt.Pz(),filt.E())
    top.SetPxPyPzE(parton.Px(),parton.Py(),parton.Pz(),parton.E())
    softdropped.SetPxPyPzE(sd.Px(),sd.Py(),sd.Pz(),sd.E())
    tau2[0]=tin.tau2
    tau3[0]=tin.tau3
    tau2_filt[0]=tin.tau2_filt
    tau3_filt[0]=tin.tau3_filt
    tau2_sd[0]=tin.tau2_sd
    tau3_sd[0]=tin.tau3_sd
    htt_tag[0]=tin.htt_tagged
    entry[0]=jentry

#    print e
    xin, yin, zin, zzin, zzzin = eta, phi, e, et, pt
#    print np.array(zin).max()
    
    
    z = raw_image(xin, yin, zin)
    zz = raw_image(xin, yin, zzin)
    zzz = raw_image(xin, yin, zzzin)
    e_aligned, et_aligned = aligned_image(z, zz)
    dump, pt_aligned = aligned_image(z, zzz)
    aaa = aligned_image_jetaxis(z, fatjet)
    if e_aligned.min() < 0:
#        print "a negative at event: ", jentry
        n_negs+=1

    if Chop:
        e_chopped = chopped_image(e_aligned,imgpix)
        et_chopped = chopped_image(et_aligned,imgpix)
        pt_chopped = chopped_image(pt_aligned,imgpix)
        bbb = chopped_image(aaa,imgpix)
    else:
        e_chopped = e_aligned
        et_chopped = et_aligned
        pt_chopped = pt_aligned
        bbb = aaa
    image_e = e_chopped.flatten()
    image_et = et_chopped.flatten()
    image_pt = pt_chopped.flatten()
    image_min = bbb.flatten()
    for j in range(len(image_e)):
        img_e[j]=image_e[j]
        img_et[j]=image_et[j]
        img_pt[j]=image_pt[j]
        img_min[j]=image_min[j]

    # # plot
    # f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2)
    # ax1.imshow(z, origin='lower',  extent=[-3.6,3.6,-180,180], aspect='auto')
    # ax2.imshow(aaa, origin='lower',  extent=[-3.6,3.6,-180,180], aspect='auto')
    # #ax1.imshow(z, origin='lower',  extent=[-1,1,-1,1], aspect='equal')
    # #ax2.imshow(zaxis, origin='lower',  extent=[-1,1,-1,1], aspect='equal')
    # ax3.imshow(b, extent=[-1,1,-1,1], aspect='equal')
    # ax4.imshow(bb, extent=[-1,1,-1,1], aspect='equal')
    # f.show()


    tout.Fill()
tout.Write()
fout.Close()

fin.Close()
