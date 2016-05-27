import numpy as np
import matplotlib.pyplot as plt
import ROOT
from ROOT import TFile, TLorentzVector, TTree
from array import array
import sys

if len(sys.argv) != 3:
    print "Provide infile and outfile"
    sys.exit()
    
#set size of final image
imgpix = int(40)

# define grid: (x,y) <-> (eta, phi), 72x72, start from quadaratic grid. The setup should match Delphes output
x = np.arange(-3.6, 3.6, 0.1)
y = np.arange(-180, 180, 5)
xi, yi = np.meshgrid(x, y)
    
def raw_image(xvalues, yvalues, zvalues):
    z = np.zeros((len(y),len(x)), dtype=float)
    for i in xrange(len(xvalues)):
        ix = np.where(xvalues[i]>=x)[0][-1]
        iy = np.where(yvalues[i]>=y)[0][-1]
        z[iy,ix] = zvalues[i]
    return z

def aligned_image(z):
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
    z = np.vstack((z,z,z))
    shift = [int(Npix[0]/2)-p1[0]-1, int(Npix[1]/2)-p1[1]-1]
    zp = scipy.ndimage.interpolation.shift(z, shift)
    zp=zp[zp.shape[0]/3:2*zp.shape[0]/3,:]
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
    zpp = scipy.ndimage.interpolation.rotate(zp, theta*180./np.pi,reshape=False, cval=10)

    # flip such that 3rd maximum has positive x
    thirdp=(np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta),  np.cos(theta)]]).dot((third-center).T)).T
    if thirdp[0,0] < -0.001:
        zpp=np.fliplr(zpp) 
    return zpp

def chopped_image(z,imgpix):
    Npix = z.shape
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
img = np.zeros(imgpix**2, dtype=float)

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
tout.Branch('img', img, 'img['+str(imgpix**2)+']/D')

entries = tin.GetEntriesFast()
for jentry in xrange(entries):
    phi = []
    eta = []
    e = []
    tin.GetEntry(jentry)
    for k in range(tin.veta.size()):
        phi.append(tin.vphi[k])
        eta.append(tin.veta[k])
        e.append(tin.ve[k])
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
   
    xin, yin, zin = phi, eta, e
    
    z = raw_image(xin, yin, zin)
    a = aligned_image(z)
    b = chopped_image(a,imgpix)
    image = b.flatten()
    for j in range(len(image)):
        img[j]=image[j]

    tout.Fill()
       
    # # plot
    # f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2)
    # ax1.imshow(z, origin='lower',  extent=[-3.6,3.6,-180,180], aspect='auto')
    # ax2.imshow(z, origin='lower',  extent=[-1,1,-1,1], aspect='equal')
    # ax3.imshow(a, extent=[-1,1,-1,1], origin='lower', aspect='equal')
    # ax4.imshow(b, extent=[-1,1,-1,1], origin='lower', aspect='equal')
    # f.show()

tout.Write()
fout.Close()

fin.Close()
