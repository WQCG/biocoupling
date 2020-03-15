## TODO - Run with a bottom bathymetry.
#!/usr/bin/python3
'''
Copyright (C) 2017-2019  Waterloo Quantitative Consulting Group, Inc.
See COPYING and LICENSE files at project root for more details.
'''

import numpy as np
from pyblitzdg import pyblitzdg as dg
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.interpolate import griddata

def positivityPreservingLimiter2D(h, hu, hv):
    Np, K = h.shape
    hmin = np.tile(np.min(h, axis=0), (Np, 1))
    hmin[hmin < 1e-3] = 1e-3

    hmean = np.tile(np.mean(h, axis=0), (Np, 1))

    theta = np.ones((Np, K))
    theta = hmean / (hmean - hmin + 1e-4)
    
    theta[theta > 1] = 1.0
    humean = np.tile(np.mean(hu, axis=0), (Np, 1))
    hvmean = np.tile(np.mean(hv, axis=0), (Np, 1))

    h  = hmean  + theta*(h  - hmean)
    hu = humean + theta*(hu - humean)
    hv = hvmean + theta*(hv - hvmean)

    return h, hu, hv


def minmod(a, b):
    soln = np.zeros(a.shape)
    for i, _ in enumerate(a):
        if a[i] < b[i] and a[i]*b[i] > 0:
            soln[i] = a[i]
        elif b[i] < a[i] and a[i]*b[i] > 0:
            soln[i] = b[i]
        else:
            soln[i] = 0.0
    
    return soln



def surfaceReconstruction(etaM, hM, etaP, hP):
    # get bed elevations
    zM = etaM - hM
    zP = etaP - hP

    dz = (zP - 0.5*minmod(zP - zM, 1e-3*np.ones(zM.shape))) - (zM + 0.5*minmod(zM-zP, -1e-3*np.ones(zM.shape)))


    # flux limit
    #etaCorrM = zP - zM - dz
    #for i,_ in enumerate(etaCorrM):
    #    if etaCorrM[i] > (etaP[i] - etaM[i]):
    #        etaCorrM[i] = etaP[i] - etaM[i]

    #    if etaCorrM[i] < 0:
    #        etaCorrM[0] = 0.0    
    
    #etaM += etaCorrM

    etaCorrP = zM - zP - dz
    for i, _ in enumerate(etaCorrP):
        if etaCorrP[i] > (etaM[i] - etaP[i]):
            etaCorrP[i] = etaM[i] - etaP[i]
            
        if etaCorrP[i] <= 0:
            etaCorrP[i] = 0.0
        else:
            etaP[i] += etaCorrP[i]


    # Get corrected bed elevation
    #zM = etaM - hM
    zP = etaP - hP

    # enforce non-negativity
    maxz = zM
    for i, _ in enumerate(zM):
        if zP[i] > zM[i]:
            maxz[i] = zP[i]

    hM = etaM - maxz
    hM[hM <= 1e-3] = 1e-3
    hP = etaP - maxz
    hP[hP <= 1e-3] = 1e-3

    return hM, hP


def write1dField(fname, array1d):
    f = open(fname, encoding="utf8", mode="w")
    for row in array1d:
        f.write(str(row) + "\n")
    f.close()


def getMaxEtaTransect(eta, x, y, vmapW):
    xFlat = x.flatten("F")
    yFlat = y.flatten("F")
    etaFlat = eta.flatten("F")

    xW = xFlat[vmapW]
    yW = yFlat[vmapW]
    etaW = etaFlat[vmapW]

    # find where max eta occurs.
    maxEtaInd = etaW.argmax()
    xMaxEta = xW[maxEtaInd]
    yMaxEta = yW[maxEtaInd]

    xTransect = np.linspace(xMaxEta, 0.0, 100)
    yTransect = np.linspace(yMaxEta, 0.0, 100)

    distTransect = np.hypot(xTransect - xMaxEta, yTransect - yMaxEta)
    etaTransect = griddata((xFlat, yFlat), etaFlat, (xTransect, yTransect), method='linear')

    return distTransect, etaTransect

def sw2dComputeFluxes(h, hu, hv, hN, g, H):
    #h equation
    F1 = hu
    G1 = hv

    # Get velocity fields
    u = hu / h
    v = hv / h

    # hu equation

    F2 = hu*u + 0.5*g*h*h
    G2 = hu*v

    # hv equation
    F3 = G2
    G3 = hv*v + 0.5*g*h*h

    # N (tracer) equation
    F4 = hN*u
    G4 = hN*v

    return ((F1,F2,F3,F4),(G1,G2,G3,G4))

def sw2dComputeRHS(h, hu, hv, hN, zx, zy, g, H, f, ctx):
    vmapM = ctx.vmapM
    vmapP = ctx.vmapP
    BCmap = ctx.BCmap
    nx = ctx.nx
    ny = ctx.ny
    rx = ctx.rx
    sx = ctx.sx
    ry = ctx.ry
    sy = ctx.sy
    Dr = ctx.Dr
    Ds = ctx.Ds
    Nfp = ctx.numFacePoints

    Lift = ctx.Lift
    Fscale = ctx.Fscale

    hC = h.flatten('F')
    huC = hu.flatten('F')
    hvC = hv.flatten('F')
    hNC = hN.flatten('F')
    nxC = nx.flatten('F')
    nyC = ny.flatten('F')
    
    mapW = BCmap[3]

    # get field values along elemental faces.
    hM = hC[vmapM]
    hP = hC[vmapP]

    eta = h - H
    etaC = eta.flatten('F')
    etaM = etaC[vmapM]
    etaP = etaC[vmapP]

    uM = huC[vmapM] / hC[vmapM]
    uP = huC[vmapP] / hC[vmapP]

    vM = hvC[vmapM] / hC[vmapM]
    vP = hvC[vmapP] / hC[vmapP]

    hNM = hNC[vmapM]
    hNP = hNC[vmapP]

    nxW = nxC[mapW]
    nyW = nyC[mapW]

    hM, hP = surfaceReconstruction(etaM, hM, etaP, hP)
    # h = np.reshape(hC, (Np, K), order='F')

    # re-form conserved transport from corrected 
    # water column heights.
    huM = hM*uM
    hvM = hM*vM

    huP = hP*uP
    hvP = hP*vP

    # set bc's (no normal flow thru the walls).
    huP[mapW] = huM[mapW] - 2*nxW*(huM[mapW]*nxW + hvM[mapW]*nyW)
    hvP[mapW] = hvM[mapW] - 2*nyW*(huM[mapW]*nxW + hvM[mapW]*nyW)

    # compute jump in states
    dh = hM - hP
    dhu = huM - huP
    dhv = hvM - hvP
    dhN = hNM - hNP

    ((F1M,F2M,F3M,F4M),(G1M,G2M,G3M,G4M)) = sw2dComputeFluxes(hM, huM, hvM, hNM, g, H)
    ((F1P,F2P,F3P,F4P),(G1P,G2P,G3P,G4P)) = sw2dComputeFluxes(hP, huP, hvP, hNP, g, H)
    ((F1,F2,F3,F4),(G1,G2,G3,G4)) = sw2dComputeFluxes(h, hu, hv, hN, g, H)

    uM = huM/hM 
    vM = hvM/hM

    uP = huP/hP
    vP = hvP/hP

    spdM = np.sqrt(uM*uM + vM*vM) + np.sqrt(g*hM)
    spdP = np.sqrt(uP*uP + vP*vP) + np.sqrt(g*hP)

    spdMax = np.max(np.array([spdM, spdP]), axis=0)

    # spdMax = np.max(spdMax)
    lam = np.reshape(spdMax, (ctx.numFacePoints, ctx.numFaces*ctx.numElements), order='F')
    lamMaxMat = np.outer(np.ones((Nfp, 1), dtype=np.float), np.max(lam, axis=0))
    spdMax = lamMaxMat.flatten('F')

    # strong form: Compute flux jump vector. (fluxM - numericalFlux ) dot nW
    dFlux1 = 0.5*((F1M - F1P)*nxC + (G1M-G1P)*nyC - spdMax*dh)
    dFlux2 = 0.5*((F2M - F2P)*nxC + (G2M-G2P)*nyC - spdMax*dhu)
    dFlux3 = 0.5*((F3M - F3P)*nxC + (G3M-G3P)*nyC - spdMax*dhv)
    dFlux4 = 0.5*((F4M - F4P)*nxC + (G4M-G4P)*nyC - spdMax*dhN)

    dFlux1Mat = np.reshape(dFlux1, (Nfp*ctx.numFaces, K), order='F')
    dFlux2Mat = np.reshape(dFlux2, (Nfp*ctx.numFaces, K), order='F')
    dFlux3Mat = np.reshape(dFlux3, (Nfp*ctx.numFaces, K), order='F')
    dFlux4Mat = np.reshape(dFlux4, (Nfp*ctx.numFaces, K), order='F')

    # Flux divergence:
    RHS1 = -(rx*np.dot(Dr, F1) + sx*np.dot(Ds, F1))
    RHS1+= -(ry*np.dot(Dr, G1) + sy*np.dot(Ds, G1))

    RHS2 = -(rx*np.dot(Dr, F2) + sx*np.dot(Ds, F2))
    RHS2+= -(ry*np.dot(Dr, G2) + sy*np.dot(Ds, G2))

    RHS3 = -(rx*np.dot(Dr, F3) + sx*np.dot(Ds, F3))
    RHS3+= -(ry*np.dot(Dr, G3) + sy*np.dot(Ds, G3))

    RHS4 = -(rx*np.dot(Dr, F4) + sx*np.dot(Ds, F4))
    RHS4+= -(ry*np.dot(Dr, G4) + sy*np.dot(Ds, G4))

    surfaceRHS1 = Fscale*dFlux1Mat
    surfaceRHS2 = Fscale*dFlux2Mat
    surfaceRHS3 = Fscale*dFlux3Mat
    surfaceRHS4 = Fscale*dFlux4Mat

    RHS1 += np.dot(Lift, surfaceRHS1)
    RHS2 += np.dot(Lift, surfaceRHS2)
    RHS3 += np.dot(Lift, surfaceRHS3)
    RHS4 += np.dot(Lift, surfaceRHS4)

    # Add source terms
    RHS2 += f*hv
    RHS3 -= f*hu

    #hbar = np.array(h.flatten())
    #hbar[vmapM] = 0.5*(hbar[vmapM] + hbar[vmapP])
    #hbar[vmapP] = hbar[vmapM]
    # hbar = np.reshape(hbar, (Np,K))
    RHS2 -= g*h*zx
    RHS3 -= g*h*zy

    return (RHS1, RHS2, RHS3, RHS4)

# Main solver:
# set scaled density jump.
drho = 1.0025 - 1.000

# compute reduced gravity
g = drho*9.81

# set f-plane Coriolis frequency.
f = 7.88e-5

c0 = np.sqrt(g*10.0)
rad = c0/f

finalTime = 24*3600
numOuts = 200
t = 0.0

meshManager = dg.MeshManager()
meshManager.readMesh('input/R_8km_circle.msh')

# Numerical parameters:
NOrder = 4

filtOrder = 4
filtCutoff = 0.6*NOrder

nodes = dg.TriangleNodesProvisioner(NOrder, meshManager)
nodes.buildFilter(filtCutoff, filtOrder)

outputter = dg.VtkOutputter(nodes)

ctx = nodes.dgContext()

x = ctx.x
y = ctx.y

BCmap = ctx.BCmap
mapW = ctx.BCmap[3]

vmapW = ctx.vmapM[mapW]

xFlat = x.flatten('F')
yFlat = y.flatten('F')

indN, indK = np.where(np.hypot(x, y) < 1.9e1)
centreIndN = indN[0]
centreIndK = indK[0]

xW = xFlat[vmapW]
yW = yFlat[vmapW]

Np = ctx.numLocalPoints
K = ctx.numElements

Filt = ctx.filter
#Filt = np.eye(Np)

#eta = -0.1*(x/8000.0)
eta = -1.0*np.exp(-((x/3000)**2 + (y/3000)**2))

distTransect, etaTransect = getMaxEtaTransect(eta, x, y, vmapW)
#write1dField("distTransect0000000.asc", distTransect)
#write1dField("etaTransect0000000.asc", etaTransect)

r = np.sqrt(x*x + y*y)

u   = np.zeros([Np, K], dtype=np.float, order='C')
v   = np.zeros([Np, K], dtype=np.float, order='C')

H = 9.5*(1-(r/8000)*(r/8000)) + .5
Dr = ctx.Dr 
Ds = ctx.Ds
rx = ctx.rx
ry = ctx.ry
sx = ctx.sx 
sy = ctx.sy

z = -H
zx = (rx*np.dot(Dr, z) + sx*np.dot(Ds, z))
zy = (ry*np.dot(Dr, z) + sy*np.dot(Ds, z))

Nrad = 2e3
Nx = 2000.0
Ny = 2500.0
# N   = np.exp(-(((x-Nx)/Nrad)**2 + ((y-Ny)/Nrad)**2))
N   = np.exp(-((y-Ny)/Nrad)**2)

#h = H + eta
h = 5*(0.5*(1 - np.tanh(x/Nrad)))
h[h < 1e-3] = 1e-3
#H = h - eta
eta = h - H
hu = h*u
hv = h*v
hN = h*N

# setup fields dictionary for outputting.
fields = dict()
fields["eta"] = eta
fields["u"] = u
fields["v"] = v
fields["N"] = N
fields["h"] = h
outputter.writeFieldsToFiles(fields, 0)

Hbar = np.mean(H)
c = np.sqrt(g*Hbar)*np.ones((Np, K))
CFL = 0.35
dt = CFL / np.max( ((NOrder+1)**2)*0.5*np.abs(ctx.Fscale.flatten('F'))*(c.flatten('F')[ctx.vmapM]  + np.sqrt(((u.flatten('F'))[ctx.vmapM])**2 + ((v.flatten('F'))[ctx.vmapM])**2)))

numSteps = int(np.ceil(finalTime/dt))
#outputInterval = int(numSteps / numOuts)
outputInterval = 10

step = 0
print("Entering main time-loop")
while t < finalTime:

    (RHS1,RHS2,RHS3,RHS4) = sw2dComputeRHS(h, hu, hv, hN, zx, zy, g, H, f, ctx)

    RHS1 = np.dot(Filt, RHS1)
    RHS2 = np.dot(Filt, RHS2)
    RHS3 = np.dot(Filt, RHS3)
    RHS4 = np.dot(Filt, RHS4)
    
    # predictor
    h1  = h + 0.5*dt*RHS1
    hu1 = hu + 0.5*dt*RHS2
    hv1 = hv + 0.5*dt*RHS3
    hN1 = hN + 0.5*dt*RHS4

    h1, hu1, hv1 = positivityPreservingLimiter2D(h1, hu1, hv1)
    h1[h1 < 1e-3] = 1e-3

    (RHS1,RHS2,RHS3,RHS4) = sw2dComputeRHS(h1, hu1, hv1, hN1, zx, zy, g, H, f, ctx)

    RHS1 = np.dot(Filt, RHS1)
    RHS2 = np.dot(Filt, RHS2)
    RHS3 = np.dot(Filt, RHS3)
    RHS4 = np.dot(Filt, RHS4)

    # corrector - Update solution
    h += dt*RHS1
    hu += dt*RHS2
    hv += dt*RHS3
    hN += dt*RHS4


    h, hu, hv = positivityPreservingLimiter2D(h, hu, hv)


    drycells = h <= 1e-3
    h[drycells] = 1e-3

    hu[drycells] = 0.0
    hv[drycells] = 0.0

    u = hu / h
    v = hv / h
    dt = CFL / np.max( ((NOrder+1)**2)*0.5*np.abs(ctx.Fscale.flatten('F'))*(c.flatten('F')[ctx.vmapM]  + np.sqrt(((u.flatten('F'))[ctx.vmapM])**2 + ((v.flatten('F'))[ctx.vmapM])**2)))


    h_max = np.max(np.abs(h))
    if h_max > 1e8  or np.isnan(h_max):
        raise Exception("A numerical instability has occurred.")

    t += dt
    step += 1

    if step % outputInterval == 0 or step == numSteps:
        print('Outputting at t=' + str(t))
        eta = h-H
        fields["eta"] = eta
        fields["u"] = hu/h
        fields["v"] = hv/h
        fields["N"] = hN/h
        fields["h"] = h
        outputter.writeFieldsToFiles(fields, step)
    
    if step == 10080 or step == 15120 or step == 5145:
        distTransect, etaTransect = getMaxEtaTransect(eta, x, y, vmapW)
        #write1dField(f"distTransect{step:07d}.asc", distTransect)
        #write1dField(f"etaTransect{step:07d}.asc", etaTransect)

