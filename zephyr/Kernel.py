"""
Contains numerical kernel for Seismic FDFD class
"""

import numpy
import scipy
import scipy.sparse
import SimPEG
from IPython.parallel import require, interactive

DEFAULT_FREESURF_BOUNDS = [False, False, False, False]
DEFAULT_PML_SIZE = 10
DEFAULT_SOLVER = scipy.sparse.linalg.splu

# ------------------------------------------------------------------------
# Parallel system setup

class SeisFDFDKernel(object):

    # source array ref

    # receiver array ref

    mesh = None
    freq = None
    Solver = lambda: None


    def __init__(self, systemConfig, **kwargs):

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
        }

        for key in initMap.keys():
            if key in systemConfig:
                if initMap[key] is None:
                    setattr(self, key, systemConfig[key])
                else:
                    setattr(self, initMap[key], systemConfig[key])

    # Model properties

    @property
    def c(self):
        return self.cR + self.cI
    @c.setter
    def c(self, value):
        self._cR = value.real
        self._cI = value.imag

    @property
    def rho(self):
        if getattr(self, '_rho', None) is None:
            self._rho = 310 * self.c**0.25
        return self._rho
    @rho.setter
    def rho(self, value):
        self._rho = value

    @property
    def Q(self):
        if getattr(self, '_Q', None) is None:
            self._Q = numpy.inf
        return self._Q
    @Q.setter
    def Q(self, value):
        self._Q = value

    @property
    def cR(self):
        return self._cR
    @cR.setter
    def cR(self, value):
        self._cR = value
    
    @property
    def cI(self):
        if self.Q is numpy.inf:
            return 0
        else:
            return -1j * self.cR / (2*self.Q)
    @cI.setter
    def cI(self, value):
        if (value == 0).all():
            self._Q = numpy.inf
        else:
            self._Q = -1j * self.cR / (2*value)

    # Modelling properties

    @property
    def nPML(self):
        if getattr(self, '_nPML', None) is None:
            self._nPML = DEFAULT_PML_SIZE
        return self._nPML
    @nPML.setter
    def nPML(self, value):
        self._nPML = value

    @property
    def freeSurf(self):
        if getattr(self, '_freeSurf', None) is None:
            self._freeSurf = DEFAULT_FREESURF_BOUNDS
        return self._freeSurf
    @freeSurf.setter
    def freeSurf(self, value):
        self._freeSurf = value

    @property
    def ky(self):
        if getattr(self, '_ky', None) is None:
            self._ky = 0.
        return self._ky
    @ky.setter
    def ky(self, value):
        self._ky = value

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
            self._Ainv = self.Solver(self.A)
        return self._Ainv

    # ------------------------------------------------------------------------
    # Matrix setup

    def _initHelmholtzNinePoint (self):
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

    def _getSource(self, sLoc):
        q = numpy.zeros(self.mesh.nN)
        qI = SimPEG.Utils.closestPoints(self.mesh, sLoc, gridLoc='N')
        q[qI] = 1

        return q

    # ------------------------------------------------------------------------
    # Externally-callable functions
    
    def forward(self, sLoc):

        q = self._getSource(sLoc)
        u = self.Ainv * q
        if self.kyweight and self.ky != 0:
            u = 2*u
        return u

    def backprop(self, sourceids):
        pass

    def misfit(self, sourceids):
        pass

@interactive
def setupSystem(systemConfig):
    global localSystem

    subSystemConfig = baseSystemConfig.copy()
    subSystemConfig.update(systemConfig)
    tag = '%f-%f'%(systemConfig['freq'], systemConfig['ky'])
    localSystem[tag] = SeisFDFDKernel(subSystemConfig)
    return tag