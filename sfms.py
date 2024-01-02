from scipy.optimize import curve_fit
import numpy as np

def line(data, a, b):
    return a*data + b

def sfmscut(m0, sfr0, threshold=-5.00E-01, m_star_min=8.0, m_star_max=10.5, m_gas_min=8.5):
    nsubs = len(m0)
    idx0  = np.arange(0, nsubs)
    non0  = ((m0   > 0.000E+00) & 
             (sfr0 > 0.000E+00) )
    m     =    m0[non0]
    sfr   =  sfr0[non0]
    idx0  =  idx0[non0]
    ssfr  = np.log10(sfr/m)
    sfr   = np.log10(sfr)
    m     = np.log10(  m)

    idxbs   = np.ones(len(m), dtype = int) * -1
    cnt     = 0
    mbrk    = 1.0200E+01
    mstp    = 2.0000E-01
    mmin    = m_star_min
    mbins   = np.arange(mmin, mbrk + mstp, mstp)
    rdgs    = []
    rdgstds = []


    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        rdg   = np.median(ssfrb)
        idxb  = (ssfrb - rdg) > threshold
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
        rdgs.append(rdg)
        rdgstds.append(np.std(ssfrb))

    rdgs       = np.array(rdgs)
    rdgstds    = np.array(rdgstds)
    mcs        = mbins[:-1] + mstp / 2.000E+00
    
    # Alex added this as a quick bug fix, no idea if it's ``correct''
    nonans = (~(np.isnan(mcs)) &
              ~(np.isnan(rdgs)) &
              ~(np.isnan(rdgs)))
        
    parms, cov = curve_fit(line, mcs[nonans], rdgs[nonans], sigma = rdgstds[nonans])
    mmin    = mbrk
    mmax    = m_star_max
    mbins   = np.arange(mmin, mmax + mstp, mstp)
    mcs     = mbins[:-1] + mstp / 2.000E+00
    ssfrlin = line(mcs, parms[0], parms[1])
        
    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        idxb  = (ssfrb - ssfrlin[i]) > threshold
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool

def trans(arr0, incl0):
    arr      = np.copy( arr0)
    incl     = np.copy(incl0)
    deg2rad  = np.pi / 1.800E+02
    incl    *= deg2rad
    arr[:,0] = -arr0[:,2] * np.sin(incl[0]) + (arr0[:,0] * np.cos(incl[1]) + arr0[:,1] * np.sin(incl[1])) * np.cos(incl[0])
    arr[:,1] = -arr0[:,0] * np.sin(incl[1]) + (arr0[:,1] * np.cos(incl[1])                                                )
    arr[:,2] =  arr0[:,2] * np.cos(incl[0]) + (arr0[:,0] * np.cos(incl[1]) + arr0[:,1] * np.sin(incl[1])) * np.sin(incl[0])
    del incl
    return arr

def calc_incl(pos0, vel0, m0, ri, ro):
    rpos = np.sqrt(pos0[:,0]**2.000E+00 +
                   pos0[:,1]**2.000E+00 +
                   pos0[:,2]**2.000E+00 )
    rpos = rpos[~np.isnan(rpos)]
    idx  = (rpos > ri) & (rpos < ro)
    pos  = pos0[idx]
    vel  = vel0[idx]
    m    =   m0[idx]
        
    hl = np.cross(pos, vel)
    L  = np.array([np.multiply(m, hl[:,0]),
                   np.multiply(m, hl[:,1]),
                   np.multiply(m, hl[:,2])])
    L  = np.transpose(L)
    L  = np.array([np.sum(L[:,0]),
                   np.sum(L[:,1]),
                   np.sum(L[:,2])])
    Lmag  = np.sqrt(L[0]**2.000E+00 +
                    L[1]**2.000E+00 +
                    L[2]**2.000E+00 )
    Lhat  = L / Lmag
    incl  = np.array([np.arccos(Lhat[2]), np.arctan2(Lhat[1], Lhat[0])])
    incl *= 1.800E+02 / np.pi
    if   incl[1]  < 0.000E+00:
         incl[1] += 3.600E+02
    elif incl[1]  > 3.600E+02:
         incl[1] -= 3.600E+02
    return incl

def center(pos0, centpos, boxsize = None):
    pos       = np.copy(pos0)
    pos[:,0] -= centpos[0]
    pos[:,1] -= centpos[1]
    pos[:,2] -= centpos[2]
    if (boxsize != None):
        pos[:,0][pos[:,0] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,0][pos[:,0] > ( boxsize / 2.000E+00)] -= boxsize
        pos[:,1][pos[:,1] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,1][pos[:,1] > ( boxsize / 2.000E+00)] -= boxsize
        pos[:,2][pos[:,2] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,2][pos[:,2] > ( boxsize / 2.000E+00)] -= boxsize
    return pos

def calc_rsfr_io(pos0, sfr0):
    fraci = 5.000E-02
    fraco = 9.000E-01
    r0    = 1.000E+01
    rpos  = np.sqrt(pos0[:,0]**2.000E+00 +
                    pos0[:,1]**2.000E+00 +
                    pos0[:,2]**2.000E+00 )
    sfr0  = sfr0[~np.isnan(rpos)]
    rpos  = rpos[~np.isnan(rpos)]
    sfr   = sfr0[np.argsort(rpos)]
    rpos  = rpos[np.argsort(rpos)]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan, np.nan
    sfrf   = np.cumsum(sfr)/sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxi   = idx0[(sfrf > fraci)]
    idxi   = idxi[0]
    rsfri  = rpos[idxi]
    dskidx = rpos < (rsfri + r0)
    sfr    =  sfr[dskidx]
    rpos   = rpos[dskidx]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan, np.nan
    sfrf   = np.cumsum(sfr) / sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxo   = idx0[(sfrf > fraco)]
    idxo   = idxo[0]
    rsfro  = rpos[idxo]
    return rsfri, rsfro

def calcrsfr(pos0, sfr0, frac = 5.000E-01, ndim = 3):
    if (ndim == 2):
        rpos = np.sqrt(pos0[:,0]**2.000E+00 + 
                       pos0[:,1]**2.000E+00 )
    if (ndim == 3):
        rpos = np.sqrt(pos0[:,0]**2.000E+00 + 
                       pos0[:,1]**2.000E+00 +
                       pos0[:,2]**2.000E+00 )    
    sfr    = sfr0[np.argsort(rpos)]
    rpos   = rpos[np.argsort(rpos)]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan
    sfrf   = np.cumsum(sfr) / sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idx    = idx0[(sfrf > frac)]
    idx    = idx[0]
    rsfr   = rpos[idx]
    return rsfr

def calczgrad(pos, m, hrho, zm9, rmax, res):
    # Search area. First 0.05 kpc, then 0.125, 0.25, 0.5, and finally 1.0 kpc
    bpass  = [5.000E-02, 1.250E-01, 2.500E-01, 5.000E-01, 1.000E+00]
    
    # Different resolutions need difference pixel sizes
    if (res == 2160): # TNG50-1 equiv.
        pixl   =  1.000E-01
        nmin   =  16 #min particles needed to be observationally equivalent
    elif (run == 1080): # TNG50-2 equiv.
        pixl   =  2.500E-01
        nmin   =  8
    elif (run == 540): # TNG50-3 equiv.
        pixl   =  5.000E-01
        nmin   =  4
    elif (run == 270): # TNG50-4 equiv.
        pixl   =  1.000E+00
        nmin   =  2
        
    dr     =  1.00E-01 #kpc
    pixa   =  pixl**2.000E+00
    sigcut =  1.000E+00
    rhocut =  1.300E-01
    mcut   =  1.000E+01**sigcut * (pixa*1.000E+06)
    
    # Create map
    pixlims = np.arange(-rmax, rmax + pixl, pixl)
    pix   = len(pixlims) - 1
    pixcs = pixlims[:-1] + (pixl / 2.000E+00)
    rs    = np.full((pix, pix), np.nan, dtype = float)
    for r in range(0, pix):
        for c in range(0, pix):
            rs[r,c] = np.sqrt(pixcs[r]**2.000E+00 + 
                              pixcs[c]**2.000E+00 )
    
    # Will not work for EAGLE
    rhoidx = hrho > rhocut
        
    # Mass map
    xym, x, y = np.histogram2d(pos[:,0], pos[:,1], weights = m, bins = [pixlims, pixlims])
    # Oxygen map
    xyo, x, y = np.histogram2d(pos[rhoidx,0], pos[rhoidx,1], weights = np.multiply(m[rhoidx], zm9[rhoidx,4]), bins = [pixlims, pixlims])
    # Hydrogen map
    xyh, x, y = np.histogram2d(pos[rhoidx,0], pos[rhoidx,1], weights = np.multiply(m[rhoidx], zm9[rhoidx,0]), bins = [pixlims, pixlims])
    xym       = np.transpose(xym)
    xyo       = np.transpose(xyo)
    xyh       = np.transpose(xyh)
    rs        = np.ravel( rs)
    xym       = np.ravel(xym)
    xyo       = np.ravel(xyo)
    xyh       = np.ravel(xyh)
    xyh[xyh < 1.000E-12] = np.nan
    
    xyoh     = xyo / xyh
    cutidx   =(xym > mcut) & (~np.isnan(xyoh))
    rs       =   rs[cutidx]
    xyoh     = xyoh[cutidx]
    xyoh     = np.log10(xyoh * (1.000E+00 / 1.600E+01)) + 1.200E+01
    rsortidx = np.argsort(rs)
    rs       =   rs[rsortidx]
    xyoh     = xyoh[rsortidx]
   
    robs   = np.arange(0.000E+00, rs[-1], dr)
    lgrad  = len(robs)
    stdrs  = np.zeros(lgrad,         dtype = float)
    medohs =  np.full(lgrad, np.nan, dtype = float)
    stdohs =  np.full(lgrad, np.nan, dtype = float)
    for i in range(0, lgrad):
        goodflag = False
        for j in range(0, len(bpass)):
            idx = ((rs > robs[i] - bpass[j]) & 
                   (rs < robs[i] + bpass[j]))
            if (np.sum(idx) >= nmin):
                goodflag = True
                break
        if (goodflag):
            stdrs[ i] =    np.std(  rs[idx])
            medohs[i] = np.median(xyoh[idx])
            stdohs[i] =    np.std(xyoh[idx])
        
    return robs, stdrs, medohs, stdohs, rs, xyoh

if __name__ == '__main__':
    print('Hello World!')