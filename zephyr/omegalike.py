import numpy as np
import SimPEG
from SimPEG import Utils
import scipy.sparse as sp

def setupClaytonEngquist (diagonals, k, dx, dz):

    beta    = 0.14
    eps     = 1e-20
    dims    = diagonals['BE'].shape

    dxx     = dx*dx
    dzz     = dz*dz
    ddiag   = np.sqrt(dx**2 + dz**2)

    # Avoid dividing by zero
    k           = k.copy()
    k[k < eps]  = eps

    # Corner terms
    babBL   = 0.25j*k[ 1, 1]*ddiag
    babTL   = 0.25j*k[-2, 1]*ddiag
    babBR   = 0.25j*k[ 1,-2]*ddiag
    babTR   = 0.25j*k[-2,-2]*ddiag

    raBL    = (1 + babBL) / (1 - babBL)
    raTL    = (1 + babTL) / (1 - babTL)
    raBR    = (1 + babBR) / (1 - babBR)
    raTR    = (1 + babTR) / (1 - babTR)

    # Temporary arrays for terms
    cc3     = np.zeros(dims, dtype=np.complex128)

    # Intermediate terms for left / right
    rr1x    = 5j/dx
    rr3x    = 2j*dzz/dx
    rr1z    = 5j/dz
    rr3z    = 2j*dxx/dz

    cc3[1:-1,   0]  = rr3x*k[ :  ,   1]
    cc3[1:-1,  -1]  = rr3x*k[ :  ,  -2]
    cc3[   0,1:-1]  = rr3z*k[   1, :  ]
    cc3[  -1,1:-1]  = rr3z*k[  -2, :  ]

    cc1             = beta * cc3
    cc1[1:-1,   0] += rr1x/k[ :  ,   1]
    cc1[1:-1,  -1] += rr1x/k[ :  ,  -2]
    cc1[   0,1:-1] += rr1z/k[   1, :  ]
    cc1[  -1,1:-1] += rr1z/k[  -2, :  ]

    aaa     = 0.5 - cc1
    abc     = aaa - 1
    bb      = 2*cc1 - 1 - cc3
    b       = bb + 2

    cc3x    = rr3x*k
    cc1x    = rr1x/k + beta*cc3x
    shiftx  = 1j*k*dz #!
    aaax    = 5 - cc1x
    abcx    = aaax - 1
    bbx     = 2*cc1x - 1 - cc3x
    bx      = bbx + 2

    # Intermediate terms for bottom / top
    cc3z    = rr3z*k
    cc1z    = rr1z/k + beta*cc3z
    shiftz  = 1j*k*dz #!
    aaaz    = 5 - cc1z
    abcz    = aaaz - 1
    bbz     = 2*cc1z - 1 - cc3z
    bz      = bbz + 2

    # Set up local variables to point at these arrays
    AD = diagonals['AD']
    DD = diagonals['DD']
    CD = diagonals['CD']
    AA = diagonals['AA']
    BE = diagonals['BE']
    CC = diagonals['CC']
    AF = diagonals['AF']
    FF = diagonals['FF']
    CF = diagonals['CF']

    # Visual key for finite-difference terms
    # (per Pratt and Worthington, 1990)
    # Modified for zero bottom left
    #
    #   This         Original
    # AF FF CF  vs.  AD DD CD
    # AA BE CC  vs.  AA BE CC
    # AD DD CD  vs.  AF FF CF

    # Center term (no corners)
    BE[   1,1:-1] = bz/dzz
    BE[  -2,1:-1] = bz/dzz
    BE[1:-1,   1] = bx/dxx
    BE[1:-1,  -2] = bx/dxx
    BE[   0,   0] = 
    BE[  -1,   0] = 
    BE[   0,  -1] = 
    BE[  -1,  -1] = 

    # Left edge zeros
    AA[ :  ,   0] = 0
    AD[ :  ,   0] = 0
    AF[ :  ,   0] = 0

    # Bottom edge zeros
    AD[   0, :  ] = 0
    DD[   0, :  ] = 0
    CD[   0, :  ] = 0

    # Top edge zeros
    AF[  -1, :  ] = 0
    FF[  -1, :  ] = 0
    CF[  -1, :  ] = 0

    # Right edge zeros
    CD[ :  ,  -1] = 0
    CC[ :  ,  -1] = 0
    CF[ :  ,  -1] = 0

    # Topedge
    be      = b/dxx
    aa      = abc/dxx
    cc      = abc/dxx
    dd      = 0
    ff      = -bb*shiftx/dxx
    ad      = 0
    cf      = -aaa*shiftx/dxx
    af      = -aaa*shiftx/dxx
    cd      = 0

# NOT CONVINCED THIS WORKS
def setupFreeSurface (diagonals, freesurf):
    keys = diagonals.keys()

    if freesurf[0]:
        for key in keys:
            if key is 'BE':
                diagonals[key][-1,:] = -1.
            else:
                diagonals[key][-1,:] = 0.

    if freesurf[1]:
        for key in keys:
            if key is 'BE':
                diagonals[key][:,-1] = -1.
            else:
                diagonals[key][:,-1] = 0.

    if freesurf[2]:
        for key in keys:
            if key is 'BE':
                diagonals[key][0,:] = -1.
            else:
                diagonals[key][0,:] = 0.

    if freesurf[3]:
        for key in keys:
            if key is 'BE':
                diagonals[key][:,0] = -1.
            else:
                diagonals[key][:,0] = 0.

def initHelmholtzNinePointCE (sc):

    # Set up SimPEG mesh
    hx = np.ones(sc['nx']) * sc['dx']
    hz = np.ones(sc['nz']) * sc['dz']
    mesh = SimPEG.Mesh.TensorMesh([hx, hz], '00')
    dims = mesh.vnN
    mAve = mesh.aveN2CC

    # Generate a complex velocity vector if Q is not infinite
    if sc['Q'] is np.inf:
        c = sc['c']
    else:
        c = sc['c'] + (-1j * sc['c'] / (2*sc['Q']))

    c = (mAve.T * c.ravel()).reshape(dims)

    # Read density model from initialization dictionary or generate
    # one using Gardner's relation
    if 'rho' in sc:
        rho = sc['rho']
    else:
        # Gardner's relation for P-velocity in m/s and density in kg/m^3
        rho = 310 * sc['c']**0.25

    rho = (mAve.T * rho.ravel()).reshape(dims)

    # fast --> slow is x --> y --> z as Fortran

    # Set up physical properties in matrices
    omega = 2 * np.pi * sc['freq']
    c = np.pad(c, pad_width=1, mode='edge')
    rho = np.pad(rho, pad_width=1, mode='edge')
    K = omega**2 / c**2
    k = np.sqrt(K) # NB: This should be modified for 2.5D case

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
        'AD':   (-1) * dims[0] + (-1), 
        'DD':   (-1) * dims[0] + ( 0),
        'CD':   (-1) * dims[0] + (+1),
        'AA':   ( 0) * dims[0] + (-1),
        'BE':   ( 0) * dims[0] + ( 0),
        'CC':   ( 0) * dims[0] + (+1),
        'AF':   (+1) * dims[0] + (-1),
        'FF':   (+1) * dims[0] + ( 0),
        'CF':   (+1) * dims[0] + (+1),
    }

    # Horizontal, vertical and diagonal geometry terms
    dx  = sc['dx']
    dz  = sc['dz']
    dxx = dx**2
    dzz = dz**2
    dxz = 2*dx*dz

    # THE DEFINITION OF TOP AND BOTTOM MAY BE WRONG

    # Buoyancies
    bMM = 1. / rho[0:-2,0:-2] # bottom left
    bME = 1. / rho[0:-2,1:-1] # bottom centre
    bMP = 1. / rho[0:-2,2:  ] # bottom centre
    bEM = 1. / rho[1:-1,0:-2] # middle left
    bEE = 1. / rho[1:-1,1:-1] # middle centre
    bEP = 1. / rho[1:-1,2:  ] # middle right
    bPM = 1. / rho[2:  ,0:-2] # top    left
    bPE = 1. / rho[2:  ,1:-1] # top    centre
    bPP = 1. / rho[2:  ,2:  ] # top    right

    # k^2
    kMM = K[0:-2,0:-2] # bottom left
    kME = K[0:-2,1:-1] # bottom centre
    kMP = K[0:-2,2:  ] # bottom centre
    kEM = K[1:-1,0:-2] # middle left
    kEE = K[1:-1,1:-1] # middle centre
    kEP = K[1:-1,2:  ] # middle right
    kPM = K[2:  ,0:-2] # top    left
    kPE = K[2:  ,1:-1] # top    centre
    kPP = K[2:  ,2:  ] # top    right

    # Reciprocal of the mass in each diagonal on the cell grid
    a1  = (bEE + bEM) / (2 * dxx)
    c1  = (bEE + bEP) / (2 * dxx)
    d1  = (bEE + bME) / (2 * dzz)
    f1  = (bEE + bPE) / (2 * dzz)
    a2  = (bEE + bMM) / (2 * dxz)
    c2  = (bEE + bPP) / (2 * dxz)
    d2  = (bEE + bMP) / (2 * dxz)
    f2  = (bEE + bPM) / (2 * dxz)

    # 9-point fd star
    acoef = 0.5461
    bcoef = 0.4539
    ccoef = 0.6248
    dcoef = 0.09381
    ecoef = 0.000001297

    # 5-point fd star
    # acoef = 1.0
    # bcoef = 0.0
    # ecoef = 0.0

    diagonals = {
        'AD':   ecoef*kMM + bcoef*a2,
        'DD':   dcoef*kME + acoef*d1,
        'CD':   ecoef*kMP + bcoef*d2,
        'AA':   dcoef*kEM + acoef*a1,
        'BE':   ccoef*kEE - acoef*(a1+c1+d1+f1) - bcoef*(a2+c2+d2+f2),
        'CC':   dcoef*kEP + acoef*c1,
        'AF':   ecoef*kPM + bcoef*f2,
        'FF':   dcoef*kPE + acoef*f1,
        'CF':   ecoef*kPP + bcoef*c2,
    }

    #setupClaytonEngquist(diagonals, k, dx, dz)

    # NOT CONVINCED THIS WORKS
    if 'freeSurf' in sc:
        setupFreeSurface(diagonals, sc['freeSurf'])

    diagonals = np.array([diagonals[key].ravel() for key in keys])
    offsets = [offsets[key] for key in keys]

    A = sp.spdiags(diagonals, offsets, mesh.nN, mesh.nN, format='csr')
    Ainv = SimPEG.SolverLU(A)

    return mesh, A, Ainv