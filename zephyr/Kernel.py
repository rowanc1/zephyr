"""
Contains numerical kernel for Seismic FDFD class
"""

import numpy
import scipy
import scipy.sparse
import SimPEG
import shutil, os, errno
from IPython.parallel import require, interactive, Reference

DEFAULT_FREESURF_BOUNDS = [False, False, False, False]
DEFAULT_PML_SIZE = 10
DEFAULT_IREG = 4
DEFAULT_SOLVER = scipy.sparse.linalg.splu

HC_KAISER = {
    1:  1.24,
    2:  2.94,
    3:  4.53,
    4:  6.31,
    5:  7.91,
    6:  9.42,
    7:  10.95,
    8:  12.53,
    9:  14.09,
    10: 14.18,
}

def KaiserWindowedSinc(ireg, offset):
    '''
    Finds 2D source terms to approximate a band-limited point source, based on

    Hicks, Graham J. (2002) Arbitrary source and receiver positioning in finite-difference
        schemes using Kaiser windowed sinc functions. Geophysics (67) 1, 156-166.

    KaiserWindowedSince(ireg, offset) --> 2D ndarray of size (2*ireg+1, 2*ireg+1)
    Input offset is the 2D offsets in fractional gridpoints between the source location and
    the nearest node on the modelling grid.
    '''
     
    from scipy.special import i0 as bessi0
    import warnings

    ireg = int(ireg)
    try:
        b = HC_KAISER.get(ireg)
    except KeyError:
        print('Kaiser windowed sinc function not implemented for half-width of %d!'%(ireg,))
        raise

    freg = 2*ireg+1

    xOffset, zOffset = offset

    # Grid from 0 to freg-1
    Zi, Xi = numpy.mgrid[:freg,:freg] 

    # Distances from source point
    dZi = (zOffset + ireg - Zi)
    dXi = (xOffset + ireg - Xi)

    # Taper terms for decay function
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tZi = numpy.nan_to_num(numpy.sqrt(1 - (dZi / ireg)**2))
        tXi = numpy.nan_to_num(numpy.sqrt(1 - (dXi / ireg)**2))
        tZi[tZi == numpy.inf] = 0
        tXi[tXi == numpy.inf] = 0

    # Actual tapers for Kaiser window
    taperZ = bessi0(b*tZi) / bessi0(b)
    taperX = bessi0(b*tXi) / bessi0(b)

    # Windowed sinc responses in Z and X
    responseZ = numpy.sinc(dZi) * taperZ
    responseX = numpy.sinc(dXi) * taperX

    # Combined 2D source response
    result = responseX * responseZ

    return result


class Rec(object):

    def __init__(self, parent, geometry, origin):

        self._parent = parent
        self._nsrc = parent.nsrc
        self._mode = geometry['mode']
        self._rec = geometry['rec']
        self._origin = origin

    def _getrec(self, i):
        if self._mode == 'fixed':
            return self._rec - self._origin
        elif self._mode == 'relative':
            return self._rec[i] - self._origin
        else:
            return None

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [self._getrec(i) for i in xrange(*index.indices(self._nsrc))]
        else:
            return self._getrec(index)

    @property
    def nrec(self):
        if getattr(self, '_nrec', None) is None:
            if self._mode == 'fixed':
                self._nrec = len(self._rec)
            else:
                self._nrec = max((len(item) for item in self._rec))

        return self._nrec

class SeisLocator25D(object):

    def __init__(self, geometry):

        x0 = geometry.get('x0', 0.)
        z0 = geometry.get('z0', 0.)
        self._origin = numpy.array([x0, 0., z0]).reshape((1,3))

        if len(geometry['src'].shape) < 2:
            self.src = geometry['src'].reshape((1,3))
        else:
            self.src = geometry['src']
        self.rec = Rec(self, geometry, self._origin)

    def __call__(self, isrc, ky):

        sloc = self.src[isrc,:].reshape((1,3)) - self._origin
        rlocs = self.rec[isrc]
        if len(rlocs.shape) < 2:
            rlocs.shape = (1,3)
        dy = abs(sloc[:,1] - rlocs[:,1])
        coeffs = numpy.cos(2*numpy.pi*ky*dy)

        return sloc[:,::2], rlocs[:,::2], coeffs

    @property
    def nsrc(self):
        return len(self.src)

    @property
    def nrec(self):
        return self.rec.nrec

# ------------------------------------------------------------------------
# Parallel system setup

class SeisFDFDKernel(object):

    # source array ref

    # receiver array ref

    mesh = None
    freq = None
    Solver = lambda: None


    def __init__(self, systemConfig, locator=None, **kwargs):

        if locator is not None:
            self._locator = locator
        else:
            self._locator = SeisLocator25D(systemConfig['geom'])

        if systemConfig.get('cache', False):
            try:
                from tempfile import mkdtemp
                from joblib import Memory
            except ImportError:
                pass
            else:
                if 'cacheDir' in systemConfig:
                    cacheDir = systemConfig['cacheDir']
                    try:
                        os.makedirs(cacheDir)
                    except OSError as e:
                        if e.errno == errno.EEXIST and os.path.isdir(cacheDir):
                            pass
                        else:
                            raise
                else:
                    cacheDir = mkdtemp()

                self._mem = Memory(cachedir=cacheDir, verbose=0)

                # Cache outputs of these methods
                self.forward = self._mem.cache(self.forward)
                self.backprop = self._mem.cache(self.backprop)

        hx = [(systemConfig['dx'], systemConfig['nx'])]
        hz = [(systemConfig['dz'], systemConfig['nz'])]
        self.mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00')

        initMap = {
        #   Argument        Rename to Property
            'c':            'cR',
            'Q':            None,
            'rho':          None,
            'nPML':         None,
            'freeSurf':     None,
            'freq':         None,
            'ky':           None,
            'kyweight':     None,
            'Solver':       None,
            'ireg':         None,
            'dx':           None,
            'dz':           None,
        }

        for key in initMap.keys():
            if key in systemConfig:
                if initMap[key] is None:
                    setattr(self, key, systemConfig[key])
                else:
                    setattr(self, initMap[key], systemConfig[key])

    def __del__(self):
        if hasattr(self, '_mem'):
            self._mem.clear()
            cacheDir = self._mem.cachedir
            del self._mem
            shutil.rmtree(cacheDir)


    # Model properties

    @property
    def c(self):
        return self.cR + self.cI
    @c.setter
    def c(self, value):
        self._cR = value.real
        self._cI = value.imag
        self._invalidateMatrix()

    @property
    def rho(self):
        if getattr(self, '_rho', None) is None:
            self._rho = 310 * self.c**0.25
        return self._rho
    @rho.setter
    def rho(self, value):
        self._rho = value
        self._invalidateMatrix()

    @property
    def Q(self):
        if getattr(self, '_Q', None) is None:
            self._Q = numpy.inf
        return self._Q
    @Q.setter
    def Q(self, value):
        self._Q = value
        self._invalidateMatrix()

    @property
    def cR(self):
        return self._cR
    @cR.setter
    def cR(self, value):
        self._cR = value
        self._invalidateMatrix()
    
    @property
    def cI(self):
        if self.Q is numpy.inf:
            return 0
        else:
            return 1j * self.cR / (2*self.Q)
    @cI.setter
    def cI(self, value):
        if (value == 0).all():
            self._Q = numpy.inf
        else:
            self._Q = 1j * self.cR / (2*value)
        self._invalidateMatrix()

    # Modelling properties

    @property
    def nPML(self):
        if getattr(self, '_nPML', None) is None:
            self._nPML = DEFAULT_PML_SIZE
        return self._nPML
    @nPML.setter
    def nPML(self, value):
        self._nPML = value
        self._invalidateMatrix()

    @property
    def freeSurf(self):
        if getattr(self, '_freeSurf', None) is None:
            self._freeSurf = DEFAULT_FREESURF_BOUNDS
        return self._freeSurf
    @freeSurf.setter
    def freeSurf(self, value):
        self._freeSurf = value
        self._invalidateMatrix()

    @property
    def ky(self):
        if getattr(self, '_ky', None) is None:
            self._ky = 0.
        return self._ky
    @ky.setter
    def ky(self, value):
        self._ky = value
        self._invalidateMatrix()

    @property
    def ireg(self):
        if getattr(self, '_ireg', None) is None:
            self._ireg = DEFAULT_IREG
        return self._ireg
    @ireg.setter
    def ireg(self, value):
        self._ireg = value
    

    # Clever matrix setup properties

    @property
    def Solver(self):
        if getattr(self, '_Solver', None) is None:
            self._Solver = SimPEG.SolverWrapD(DEFAULT_SOLVER)
        return self._Solver
    @Solver.setter
    def Solver(self, value):
        self._Solver = value

    @property
    def A(self):
        if getattr(self, '_A', None) is None:
            self._A = self._initHelmholtzNinePoint()
        return self._A

    @property
    def Ainv(self):
        if getattr(self, '_Ainv', None) is None:
            self._mfact()
        return self._Ainv

    def _invalidateMatrix(self):
        if getattr(self, '_A', None) is not None:
            del(self._A)
        if getattr(self, '_Ainv', None) is not None:
            del(self._Ainv)
        if getattr(self, '_mem', None) is not None:
            self._mem.clear()

    # ------------------------------------------------------------------------
    # Matrix setup

    def _mfact(self):
        self._Ainv = self.Solver(self.A)

    def _initHelmholtzNinePoint(self):
        """
        An attempt to reproduce the finite-difference stencil and the
        general behaviour of OMEGA by Pratt et al. The stencil is a 9-point
        second-order version based on work by a number of people in the mid-90s
        including Ivan Stekl. The boundary conditions are based on the PML
        implementation by Steve Roecker in fdfdpml.f.
        """

        # Set up SimPEG mesh
        dims = (self.mesh.nNy, self.mesh.nNx)
        mAve = self.mesh.aveN2CC

        c = (mAve.T * self.c.ravel()).reshape(dims)
        rho = (mAve.T * self.rho.ravel()).reshape(dims)

        # fast --> slow is x --> y --> z as Fortran

        # Set up physical properties in matrices with padding
        omega   = 2 * numpy.pi * self.freq 
        cPad    = numpy.pad(c, pad_width=1, mode='edge')
        rhoPad  = numpy.pad(rho, pad_width=1, mode='edge')

        aky = 2*numpy.pi*self.ky

        # Model parameter M
        K = ((omega**2 / cPad**2) - aky**2) / rhoPad

        # Horizontal, vertical and diagonal geometry terms
        dx  = self.mesh.hx[0]
        dz  = self.mesh.hy[0]
        dxx = dx**2
        dzz = dz**2
        dxz = dx*dz
        dd  = numpy.sqrt(dxz)

        # PML decay terms
        # NB: Arrays are padded later, but 'c' in these lines
        #     comes from the original (un-padded) version

        nPML    = self.nPML

        pmldx   = dx*(nPML - 1)
        pmldz   = dz*(nPML - 1)
        pmlr    = 1e-3
        pmlfx   = 3.0 * numpy.log(1/pmlr)/(2*pmldx**3)
        pmlfz   = 3.0 * numpy.log(1/pmlr)/(2*pmldz**3)

        dpmlx   = numpy.zeros(dims, dtype=numpy.complex128)
        dpmlz   = numpy.zeros(dims, dtype=numpy.complex128)
        isnx    = numpy.zeros(dims, dtype=numpy.float64)
        isnz    = numpy.zeros(dims, dtype=numpy.float64)

        # Only enable PML if the free surface isn't set

        freeSurf = self.freeSurf

        if freeSurf[0]:    
            isnz[-nPML:,:] = -1 # Top

        if freeSurf[1]:
            isnx[:,-nPML:] = -1 # Right Side

        if freeSurf[2]:
            isnz[:nPML,:] = 1 # Bottom

        if freeSurf[3]:
            isnx[:,:nPML] = 1 # Left side

        dpmlx[:,:nPML] = (numpy.arange(nPML, 0, -1)*dx).reshape((1,nPML))
        dpmlx[:,-nPML:] = (numpy.arange(1, nPML+1, 1)*dx).reshape((1,nPML))
        dnx     = pmlfx*c*dpmlx**2
        ddnx    = 2*pmlfx*c*dpmlx
        denx    = dnx + 1j*omega
        r1x     = 1j*omega / denx
        r1xsq   = r1x**2
        r2x     = isnx*r1xsq*ddnx/denx

        dpmlz[:nPML,:] = (numpy.arange(nPML, 0, -1)*dz).reshape((nPML,1))
        dpmlz[-nPML:,:] = (numpy.arange(1, nPML+1, 1)*dz).reshape((nPML,1))
        dnz     = pmlfz*c*dpmlz**2
        ddnz    = 2*pmlfz*c*dpmlz
        denz    = dnz + 1j*omega
        r1z     = 1j*omega / denz
        r1zsq   = r1z**2
        r2z     = isnz*r1zsq*ddnz/denz

        # Visual key for finite-difference terms
        # (per Pratt and Worthington, 1990)
        #
        #   This         Original
        # AF FF CF  vs.  AD DD CD
        # AA BE CC  vs.  AA BE CC
        # AD DD CD  vs.  AF FF CF

        # Set of keys to index the dictionaries
        keys = ['AD', 'DD', 'CD', 'AA', 'BE', 'CC', 'AF', 'FF', 'CF']

        # Diagonal offsets for the sparse matrix formation
        offsets = {
            'AD':   (-1) * dims[1] + (-1), 
            'DD':   (-1) * dims[1] + ( 0),
            'CD':   (-1) * dims[1] + (+1),
            'AA':   ( 0) * dims[1] + (-1),
            'BE':   ( 0) * dims[1] + ( 0),
            'CC':   ( 0) * dims[1] + (+1),
            'AF':   (+1) * dims[1] + (-1),
            'FF':   (+1) * dims[1] + ( 0),
            'CF':   (+1) * dims[1] + (+1),
        }

        # Buoyancies
        bMM = 1. / rhoPad[0:-2,0:-2] # bottom left
        bME = 1. / rhoPad[0:-2,1:-1] # bottom centre
        bMP = 1. / rhoPad[0:-2,2:  ] # bottom centre
        bEM = 1. / rhoPad[1:-1,0:-2] # middle left
        bEE = 1. / rhoPad[1:-1,1:-1] # middle centre
        bEP = 1. / rhoPad[1:-1,2:  ] # middle right
        bPM = 1. / rhoPad[2:  ,0:-2] # top    left
        bPE = 1. / rhoPad[2:  ,1:-1] # top    centre
        bPP = 1. / rhoPad[2:  ,2:  ] # top    right

        # Initialize averaged buoyancies on most of the grid
        bMM = (bEE + bMM) / 2 # a2
        bME = (bEE + bME) / 2 # d1
        bMP = (bEE + bMP) / 2 # d2
        bEM = (bEE + bEM) / 2 # a1
        # ... middle
        bEP = (bEE + bEP) / 2 # c1
        bPM = (bEE + bPM) / 2 # f2
        bPE = (bEE + bPE) / 2 # f1
        bPP = (bEE + bPP) / 2 # c2

        # Reset the buoyancies on the outside edges
        bMM[ 0, :] = bEE[ 0, :]
        bMM[ :, 0] = bEE[ :, 0]
        bME[ 0, :] = bEE[ 0, :]
        bMP[ 0, :] = bEE[ 0, :]
        bMP[ :,-1] = bEE[ :,-1]
        bEM[ :, 0] = bEE[ :, 0]
        bEP[ :,-1] = bEE[ :,-1]
        bPM[-1, :] = bEE[-1, :]
        bPM[ :, 0] = bEE[ :, 0]
        bPE[-1, :] = bEE[-1, :]
        bPP[-1, :] = bEE[-1, :]
        bPP[ :,-1] = bEE[ :,-1]

        # K = omega^2/(c^2 . rho)
        kMM = K[0:-2,0:-2] # bottom left
        kME = K[0:-2,1:-1] # bottom centre
        kMP = K[0:-2,2:  ] # bottom centre
        kEM = K[1:-1,0:-2] # middle left
        kEE = K[1:-1,1:-1] # middle centre
        kEP = K[1:-1,2:  ] # middle right
        kPM = K[2:  ,0:-2] # top    left
        kPE = K[2:  ,1:-1] # top    centre
        kPP = K[2:  ,2:  ] # top    right

        # 9-point fd star
        acoef   = 0.5461
        bcoef   = 0.4539
        ccoef   = 0.6248
        dcoef   = 0.09381
        ecoef   = 0.000001297

        # 5-point fd star
        # acoef = 1.0
        # bcoef = 0.0
        # ecoef = 0.0

        # NB: bPM and bMP here are switched relative to S. Roecker's version
        #     in OMEGA. This is because the labelling herein is always ?ZX.

        diagonals = {
            'AD':   ecoef*kMM
                    + bcoef*bMM*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
            'DD':   dcoef*kME
                    + acoef*bME*(r1zsq/dz - r2z/2)/dz
                    + bcoef*(r1zsq-r1xsq)*(bMP+bMM)/(4*dxz),
            'CD':   ecoef*kMP
                    + bcoef*bMP*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
            'AA':   dcoef*kEM
                    + acoef*bEM*(r1xsq/dx - r2x/2)/dx
                    + bcoef*(r1xsq-r1zsq)*(bPM+bMM)/(4*dxz),
            'BE':   ccoef*kEE
                    + acoef*(r2x*(bEM-bEP)/(2*dx) + r2z*(bME-bPE)/(2*dz) - r1xsq*(bEM+bEP)/dxx - r1zsq*(bME+bPE)/dzz)
                    + bcoef*(((r2x+r2z)*(bMM-bPP) + (r2z-r2x)*(bMP-bPM))/(4*dd) - (r1xsq+r1zsq)*(bMM+bPP+bPM+bMP)/(4*dxz)),
            'CC':   dcoef*kEP
                    + acoef*bEP*(r1xsq/dx + r2x/2)/dx
                    + bcoef*(r1xsq-r1zsq)*(bMP+bPP)/(4*dxz),
            'AF':   ecoef*kPM
                    + bcoef*bPM*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
            'FF':   dcoef*kPE
                    + acoef*bPE*(r1zsq/dz - r2z/2)/dz
                    + bcoef*(r1zsq-r1xsq)*(bPM+bPP)/(4*dxz),
            'CF':   ecoef*kPP
                    + bcoef*bPP*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
        }

        self._setupBoundary(diagonals, freeSurf)

        diagonals = numpy.array([diagonals[key].ravel() for key in keys])
        offsets = [offsets[key] for key in keys]

        A = scipy.sparse.spdiags(diagonals, offsets, self.mesh.nN, self.mesh.nN, format='csr')

        return A

    def _setupBoundary(self, diagonals, freeSurf):
        """
        Function to set up boundary regions for the Seismic FDFD problem
        using the 9-point finite-difference stencil from OMEGA/FULLWV.
        """

        keys = diagonals.keys()
        pickDiag = lambda x: -1. if freeSurf[x] else 1.

        # Left
        for key in keys:
            if key is 'BE':
                diagonals[key][:,0] = pickDiag(3)
            else:
                diagonals[key][:,0] = 0.

        # Right
        for key in keys:
            if key is 'BE':
                diagonals[key][:,-1] = pickDiag(1)
            else:
                diagonals[key][:,-1] = 0.

        # Bottom
        for key in keys:
            if key is 'BE':
                diagonals[key][0,:] = pickDiag(2)
            else:
                diagonals[key][0,:] = 0.

        # Top
        for key in keys:
            if key is 'BE':
                diagonals[key][-1,:] = pickDiag(0)
            else:
                diagonals[key][-1,:] = 0.

    # Quasi-functional attempt -----------------------------------------------
    #
    def _srcVec(self, sLocs, terms):

        q = numpy.zeros((self.mesh.nNy, self.mesh.nNx), dtype=numpy.complex128)
        srcScale = -self.dx*self.dz

        if self.ireg == 0:
            # Closest source point
            q = q.ravel()

            for i in xrange(len(sLocs)):
                qI = SimPEG.Utils.closestPoints(self.mesh, sLocs[i], gridLoc='N')
                q[qI] += terms[i]/srcScale

        else:
            # Kaiser windowed sinc function

            freg = 2*self.ireg+1
            q = numpy.pad(q, self.ireg, mode='constant')

            for i in xrange(len(sLocs)):
                qI = SimPEG.Utils.closestPoints(self.mesh, sLocs[i], gridLoc='N')
                Zi, Xi = (qI / self.mesh.nNx, numpy.mod(qI, self.mesh.nNx))
                offset = (sLocs[i][0] - Xi * self.dx, sLocs[i][1] - Zi * self.dz)
                sourceRegion = KaiserWindowedSinc(self.ireg, offset)
                q[Zi:Zi+freg,Xi:Xi+freg] += terms[i] * sourceRegion / srcScale

            # Mirror and flip sign on terms that cross the free-surface boundary
            if self.freeSurf[0]:
                q[self.ireg:2*self.ireg,:]      -= numpy.flipud(q[:self.ireg,:])    # Top
            if self.freeSurf[1]:
                q[:,-2*self.ireg:-self.ireg]    -= numpy.fliplr(q[:,-self.ireg:])   # Right
            if self.freeSurf[2]:
                q[-2*self.ireg:-self.ireg,:]    -= numpy.flipud(q[-self.ireg:,:])   # Bottom
            if self.freeSurf[3]:
                q[:,self.ireg:2*self.ireg]      -= numpy.fliplr(q[:,:self.ireg])    # Left

            # Cut off edges
            q = q[self.ireg:-self.ireg,self.ireg:-self.ireg].ravel()

        return q

    def _srcTerm(self, sLocs, individual=True, terms=1):

        if individual and len(sLocs) > 1:
            result = []
            for i in xrange(len(sLocs)):
                result.append(self._srcVec([sLocs[i] if hasattr(sLocs, '__contains__') else sLocs], [terms[i]] if hasattr(terms, '__contains__') else [terms]))
        else:
            result = self._srcVec(sLocs if hasattr(sLocs, '__contains__') else [sLocs], terms if hasattr(terms, '__contains__') else [terms])

        return result 
    #
    # Quasi-functional attempt -----------------------------------------------

    # ------------------------------------------------------------------------
    # Externally-callable functions

    def clear(self):
        self._invalidateMatrix()
    
    # What about @caching decorators?
    def forward(self, isrc, dOnly=True, sterm=1.):

        sloc, rlocs, coeffs = self._locator(isrc, self.ky)

        q = self._srcTerm(sloc, individual=True, terms=sterm)
        u = self.Ainv * q

        d = numpy.array([numpy.dot(u,qr) for qr in self._srcTerm(rlocs, individual=True, terms=coeffs)])

        if dOnly:
            return d
        else:
            return u, d

    def backprop(self, isrc, dresid=1.):
        
        sloc, rlocs, coeffs = self._locator(isrc, self.ky)

        qr = self._srcTerm(rlocs, individual=False, terms=dresid*coeffs)

        u = self.Ainv * qr

        return u

    # def gradient(self, isrc, sterm, dresid):

    #     uF, d = self.forward(isrc, False, sterm)
    #     uB = self.backprop(isrc, dresid)

    #     return uF * uB

