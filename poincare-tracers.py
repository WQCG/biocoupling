#!/usr/bin/python3
'''
Copyright (C) 2017-2019  Waterloo Quantitative Consulting Group, Inc.
See COPYING and LICENSE files at project root for more details.
'''

import numpy as np
import pyblitzdg as dg
from scipy.special import jv as besselj
from scipy.special import iv as besseli
from scipy.optimize import root_scalar as root

# a: radius of circular lake (m)
# c0: Long wave speed (m/s)
# f: Coriolis parameter (1/s) 
# azimuthal_m: azimuthal mode number (-)
# radial_n: radial mode number (-)
def calculateEigenPair(a, c0, f, azimuthal_m, radial_n, mode_type, sign=1):
    if mode_type not in {"Kelvin", "Poincare"}:
        raise ValueError("Unknown mode_type: '" + mode_type + "'")
    
    if sign not in {-1, 1}:
        raise ValueError("Invalid sign: '" + str(sign) + "'")

    if mode_type == 'Kelvin' and sign == 1:
        raise ValueError("Invalid parameter selection. Kelvin modes with clockwise phase propagation do not exist")

    bracket_factor = 2e-2
    eps = 1e-7

    PoincareEqn = lambda sig: (a/np.sqrt(c0**2 / (sig**2 - f**2)))*((1/azimuthal_m)*besselj(azimuthal_m - 1 , a/np.sqrt(c0**2 / (sig**2 - f**2))) /
        (a/np.sqrt(c0**2 / (sig**2 - f**2)))* besselj(azimuthal_m, a/np.sqrt(c0**2 / (sig**2 - f**2))) ) - 1 + (f/sig)

    KelvinEqn = lambda sig: (a/np.sqrt(c0**2 / (f**2 - sig**2)))*((1/azimuthal_m)*besseli(azimuthal_m - 1 , a/np.sqrt(c0**2 / (f**2 - sig**2))) /
        (a/np.sqrt(c0**2 / (f**2 - sig**2)))* besseli(azimuthal_m, a/np.sqrt(c0**2 / (f**2 - sig**2))) ) - 1 + (f/sig)

    # Calculate sigma
    if mode_type == 'Poincare' and sign == 1:
        sig = f
        for n in range(0, radial_n):
            rt = root(PoincareEqn, method="bisect", bracket=[(1+eps)*sig, (1+bracket_factor)*sig])
            if rt.root is None:
                print("Did not find an eigenvalue in expected range for Poincare mode with n=" + str(n))
            sig = rt.root
    elif mode_type == 'Poincare' and sign == -1:
        sig = -f
        for n in range(0, radial_n):
            rt = root(PoincareEqn, method="bisect", bracket=[(1+bracket_factor)*sig, (1+eps)*sig])
            if rt.root is None:
                print("Did not find an eigenvalue in expected range for Poincare mode with n=" + str(n))
            sig = rt.root
    else:
        sig = -f
        for n in range(0, radial_n):  # Can also restrict n's on kelvin mode using Csanady's restriction on S.
            rt = root(KelvinEqn, method="bisect", bracket=[(1-eps)*sig, -eps])
            if rt.root is None:
                print("Did not find an eigenvalue in expected range for Kelvin mode with n=" + str(n))
            sig = rt.root

    Theta = lambda theta:  np.exp(1j*azimuthal_m*theta)
    if mode_type == "Kelvin":
        R = lambda r: besseli(azimuthal_m, np.sqrt((f**2 - sig**2) / c0**2)*r)
    else:
        #Poincare
        R = lambda r: besselj(azimuthal_m, np.sqrt((sig**2 - f**2) / c0**2)*r)

    eigenfunction = lambda r, theta: R(r) * Theta(theta) 
    return (sig, (lambda r, theta: eigenfunction(r, theta)) )

def flowSolution(t, sig, phi, ctx, r, theta):
    # eta = np.exp(1j*azimuthal_m*tt)*np.exp(1j*sig0*t)*besselj(azimuthal_m, np.sqrt((sig0**2 - f**2) / c0**2)*(rr))
    eta = np.exp(1j*sig*t)*phi(r, theta)
    scal = np.max(np.abs(eta).flatten()) 
    eta_x = ctx.rx*np.dot(ctx.Dr, eta) + ctx.sx*np.dot(ctx.Ds, eta)
    eta_y = ctx.ry*np.dot(ctx.Dr, eta) + ctx.sy*np.dot(ctx.Ds, eta)
    eta = np.real(eta) / scal
    u = np.real((-g / (f**2 - sig0**2))*(1j*sig0*eta_x + f*eta_y)) / scal
    v = np.real((-g / (f**2 - sig0**2))*(1j*sig0*eta_y - f*eta_x)) / scal

    return (eta, u, v)

def computeRHS(h, u, v, hN, hP, hZ, hD, Filt):
    hN_fluxX = hN*u
    hN_fluxY = hN*v

    hP_fluxX = hP*u
    hP_fluxY = hP*v

    hZ_fluxX = hZ*u
    hZ_fluxY = hZ*v

    hD_fluxX = hD*u
    hD_fluxY = hD*v

    hN_fluxX_x = rx*np.dot(Dr, hN_fluxX) + sx*np.dot(Ds, hN_fluxX)
    hN_fluxY_y = ry*np.dot(Dr, hN_fluxY) + sy*np.dot(Ds, hN_fluxY)

    hP_fluxX_x = rx*np.dot(Dr, hP_fluxX) + sx*np.dot(Ds, hP_fluxX)
    hP_fluxY_y = ry*np.dot(Dr, hP_fluxY) + sy*np.dot(Ds, hP_fluxY)

    hZ_fluxX_x = rx*np.dot(Dr, hZ_fluxX) + sx*np.dot(Ds, hZ_fluxX)
    hZ_fluxY_y = ry*np.dot(Dr, hZ_fluxY) + sy*np.dot(Ds, hZ_fluxY)

    hD_fluxX_x = rx*np.dot(Dr, hD_fluxX) + sx*np.dot(Ds, hD_fluxX)
    hD_fluxY_y = ry*np.dot(Dr, hD_fluxY) + sy*np.dot(Ds, hD_fluxY)

    RHShN = -(hN_fluxX_x + hN_fluxY_y)
    RHShP = -(hP_fluxX_x + hP_fluxY_y)
    RHShZ = -(hZ_fluxX_x + hZ_fluxY_y)
    RHShD = -(hD_fluxX_x + hD_fluxY_y)
    
    hF = h.flatten('F')
    uF = u.flatten('F')
    vF = v.flatten('F')

    hN_fluxX_F = hN_fluxX.flatten('F')
    hN_fluxY_F = hN_fluxY.flatten('F')

    hP_fluxX_F = hP_fluxX.flatten('F')
    hP_fluxY_F = hP_fluxY.flatten('F')

    hZ_fluxX_F = hZ_fluxX.flatten('F')
    hZ_fluxY_F = hZ_fluxY.flatten('F')

    hD_fluxX_F = hD_fluxX.flatten('F')
    hD_fluxY_F = hD_fluxY.flatten('F')

    uM = uF[vmapM]
    vM = vF[vmapM]

    hN_fluxX_M = hN_fluxX_F[vmapM]
    hN_fluxY_M = hN_fluxY_F[vmapM]
    hP_fluxX_M = hP_fluxX_F[vmapM]
    hP_fluxY_M = hP_fluxY_F[vmapM]
    hZ_fluxX_M = hZ_fluxX_F[vmapM]
    hZ_fluxY_M = hZ_fluxY_F[vmapM]
    hD_fluxX_M = hD_fluxX_F[vmapM]
    hD_fluxY_M = hD_fluxY_F[vmapM]

    uP = uF[vmapP]
    vP = vF[vmapP]

    hN_fluxX_P = hN_fluxX_F[vmapP]
    hN_fluxY_P = hN_fluxY_F[vmapP]
    hP_fluxX_P = hP_fluxX_F[vmapP]
    hP_fluxY_P = hP_fluxY_F[vmapP]
    hZ_fluxX_P = hZ_fluxX_F[vmapP]
    hZ_fluxY_P = hZ_fluxY_F[vmapP]
    hD_fluxX_P = hD_fluxX_F[vmapP]
    hD_fluxY_P = hD_fluxY_F[vmapP]

    spdM = np.sqrt(uM*uM + vM*vM) + np.sqrt(g*hF[vmapP])
    spdP = np.sqrt(uP*uP + vP*vP) + np.sqrt(g*hF[vmapP])

    spdMax = np.max(np.array([spdM, spdP]), axis=0)

    nxF = ctx.nx.flatten('F')
    nyF = ctx.ny.flatten('F')

    hN_F = hN.flatten('F')
    hP_F = hP.flatten('F')
    hZ_F = hZ.flatten('F')
    hD_F = hD.flatten('F')

    dhN = hN_F[vmapM] - hN_F[vmapP]
    dhP = hP_F[vmapM] - hP_F[vmapP]
    dhZ = hZ_F[vmapM] - hZ_F[vmapP]
    dhD = hD_F[vmapM] - hD_F[vmapP]

    # spdMax = np.max(spdMax)
    lam = np.reshape(spdMax, (ctx.numFacePoints, ctx.numFaces*ctx.numElements), order='F')
    lamMaxMat = np.outer(np.ones((ctx.numFacePoints, 1), dtype=np.float), np.max(lam, axis=0))
    spdMax = lamMaxMat.flatten('F')

    # strong form: Compute flux jump vector. (fluxM - numericalFlux ) dot n
    dFlux1 = 0.5*((hN_fluxX_M - hN_fluxX_P)*nxF + (hN_fluxY_M-hN_fluxY_P)*nyF - spdMax*dhN)
    dFlux2 = 0.5*((hP_fluxX_M - hP_fluxX_P)*nxF + (hP_fluxY_M-hP_fluxY_P)*nyF - spdMax*dhP)
    dFlux3 = 0.5*((hZ_fluxX_M - hZ_fluxX_P)*nxF + (hZ_fluxY_M-hZ_fluxY_P)*nyF - spdMax*dhZ)
    dFlux4 = 0.5*((hD_fluxX_M - hD_fluxX_P)*nxF + (hD_fluxY_M-hD_fluxY_P)*nyF - spdMax*dhD)

    dFlux1Mat = np.reshape(dFlux1, (ctx.numFacePoints*ctx.numFaces, K), order='F')
    dFlux2Mat = np.reshape(dFlux2, (ctx.numFacePoints*ctx.numFaces, K), order='F')
    dFlux3Mat = np.reshape(dFlux3, (ctx.numFacePoints*ctx.numFaces, K), order='F')
    dFlux4Mat = np.reshape(dFlux4, (ctx.numFacePoints*ctx.numFaces, K), order='F')

    surfaceRHS1 = ctx.Fscale*dFlux1Mat
    surfaceRHS2 = ctx.Fscale*dFlux2Mat
    surfaceRHS3 = ctx.Fscale*dFlux3Mat
    surfaceRHS4 = ctx.Fscale*dFlux4Mat

    RHShN += np.dot(ctx.Lift, surfaceRHS1)
    RHShP += np.dot(ctx.Lift, surfaceRHS2)
    RHShZ += np.dot(ctx.Lift, surfaceRHS3)
    RHShD += np.dot(ctx.Lift, surfaceRHS4)

    RHShN = np.dot(Filt, RHShN)
    RHShP = np.dot(Filt, RHShP)
    RHShZ = np.dot(Filt, RHShZ)
    RHShD = np.dot(Filt, RHShD)

    return (RHShN, RHShP, RHShZ, RHShD)

# Main solver:
if __name__ == '__main__':
    g = 9.81
    t = 0.0

    # ALL the possible parameters
    Nt	= 2.0		# mmol N/m^3
    mu	= 2.0		# / day
    kN   = 1.0		# mmol N/m^3
    mP   = 0.1		# /day
    mZ   = 0.2		# /day
    alph = 0.7		# efficiency to Z
    beta = 0.3      # efficiency to N
    a    = 1.0		# mmol N/m^3
    Imax = 1.5   	# /day
    b    = 1.0		# mmol N/m^3
    c    = 1.5		# mmol N/m^3
    d    = 1.5      # /day
    kp   = 0.7		# ????
    k    = 0.05

    waveScale = 5e-2 # 5 cm wave height scaling
    timeScale = 86400 # unit time = 1 day

    myI = lambda P: (P/(kp+P))*Imax # our predation function
    dP  = lambda N,P,Z,D: (N/(kN+N))*mu*P - mP*P - myI(P)*Z
    dZ  = lambda N,P,Z,D: alph*myI(P)*Z - mZ*Z
    dD  = lambda N,P,Z,D: mP*P + beta*myI(P)*Z + mZ*Z - k*D
    dN  = lambda N,P,Z,D: -(N/(kN+N))*mu*P + (1-alph-beta)*myI(P)*Z + k*D

    meshManager = dg.MeshManager()
    meshManager.readMesh('./input/csanady_circle.msh')

    # Numerical parameters:
    NOrder = 8
    CFL = 0.9

    filtOrder = 4
    filtCutoff = 0.65*NOrder

    nodes = dg.TriangleNodesProvisioner(NOrder, meshManager)
    nodes.buildFilter(filtCutoff, filtOrder)

    outputter = dg.VtkOutputter(nodes)

    ctx = nodes.dgContext()

    x = ctx.x
    y = ctx.y
    rx = ctx.rx
    ry = ctx.ry
    sx = ctx.sx
    sy = ctx.sy
    Dr = ctx.Dr
    Ds = ctx.Ds

    Np = ctx.numLocalPoints
    K = ctx.numElements

    Filt = ctx.filter

    f = 1e-4
    a = 67.5e3
    drho = 1.74
    rho0 = 1000
    g = 9.81
    H1 = 15
    H2 = 60
    gprime = (drho/rho0)*g
    H = H1*H2/(H1+H2)
    # H = H1 + H2
    c0 = np.sqrt(gprime*H)
    lam1 = 1e-4
    lam2 = 1e-4

    azimuthal_m = 1
    radial_n = 1
    (sig0, phi0) = calculateEigenPair(a, c0, f, azimuthal_m, radial_n, "Poincare", 1)

    print("sig / f: " + str(sig0 / f))

    tt = np.arctan2(y, x).flatten('F')
    tt[tt < 0] = 2*np.pi + tt[tt < 0]
    tt = np.reshape(tt, (Np, K), order='F')
    rr = np.sqrt(x**2 + y**2)

    step = 0
    t = 0.0

    # Tracers (N, P, D, Z)
    N = 1*np.exp(-( ((x-0.15*a)/(0.15*a))**2  + ((y - 0.15*a)/(0.15*a))**2))
    P = 1*np.exp(-( ((x-0.15*a)/(0.2*a))**2  + ((y - 0.15*a)/(0.1*a))**2))
    Z = 0.1*np.exp(-( ((x-0.15*a)/(0.05*a))**2  + ((y - 0.15*a)/(0.05*a))**2))
    D = 0.5*np.exp(-( ((x-0.15*a)/(0.30*a))**2  + ((y - 0.15*a)/(0.30*a))**2))

    vmapM = ctx.vmapM
    vmapP = ctx.vmapP
    mapW = ctx.BCmap[3]
    vmapW = vmapM[mapW]

    finalTime = np.abs(2*np.pi / sig0) / timeScale
    print("T final: " + str(finalTime) + " days")

    sig0 *= timeScale
    Nn = N
    Pn = P
    Zn = Z
    Dn = D
    (eta, u, v) = flowSolution(t, sig0, phi0, ctx, rr, tt)
    eta *= waveScale
    u   *= waveScale * timeScale
    v   *= waveScale * timeScale
        
    esteps=10
    while t < finalTime:
        if (step % 25) == 0:
            print('outputting at t=' + str(t))
            # setup fields dictionary for outputting.
            fields = dict()
            fields["eta"] = eta
            fields["u"] = u
            fields["v"] = v
            fields["N"] = N
            fields["P"] = P
            fields["Z"] = Z
            fields["D"] = D
            outputter.writeFieldsToFiles(fields, step)
        
        dt = CFL / np.max( ((NOrder+1)**2)*0.5*np.abs(ctx.Fscale.flatten('F'))*(c0*timeScale  + np.sqrt(((u.flatten('F'))[vmapM])**2 + ((v.flatten('F'))[vmapM])**2)))
        
        dte=dt/esteps

        h = H + eta
        hNn = h*Nn
        hPn = h*Pn
        hZn = h*Zn
        hDn = h*Dn
        (RHShNn, RHShPn, RHShZn, RHShDn) = computeRHS(h, u, v, hNn, hPn, hZn, hDn, Filt)

        # Get the explicit midpoint.
        hNhat = hNn + 0.5*dt*RHShNn
        hPhat = hPn + 0.5*dt*RHShPn
        hZhat = hZn + 0.5*dt*RHShZn
        hDhat = hDn + 0.5*dt*RHShDn

        (etahat, uhat, vhat) = flowSolution(t+0.5*dt, sig0, phi0, ctx, rr, tt)
        etahat *= waveScale
        uhat   *= waveScale * timeScale
        vhat   *= waveScale * timeScale
    
        hhat = H + etahat
        (RHShNhat, RHShPhat, RHShZhat, RHShDhat) = computeRHS(hhat, uhat, vhat, hNhat, hPhat, hZhat, hDhat, Filt)

        # 'Improved Euler' corrector.
        hNnp1 = hNn + dt*RHShNhat
        hPnp1 = hPn + dt*RHShPhat
        hZnp1 = hZn + dt*RHShZhat
        hDnp1 = hDn + dt*RHShDhat

        (etanp1, unp1, vnp1) = flowSolution(t+dt, sig0, phi0, ctx, rr, tt)
        etanp1 *= waveScale
        unp1   *= waveScale * timeScale
        vnp1   *= waveScale * timeScale
    
        hnp1 = H + etanp1
        N = hNnp1 / hnp1
        P = hPnp1 / hnp1
        Z = hZnp1 / hnp1
        D = hDnp1 / hnp1
        
        # subcycled Euler (once, for now)
        for ei in range(0, esteps):
            N+= dte*dN(N, P, Z, D)
            P+= dte*dP(N, P, Z, D)
            Z+= dte*dZ(N, P, Z, D)
            D+= dte*dD(N, P, Z, D)

        N_max = np.max(np.abs(N))
        if N_max > 1e8  or np.isnan(N_max):
            raise Exception("A numerical instability has occurred.")

        t += dt
        step += 1

        Nn = N
        Pn = P
        Zn = Z
        Dn = D
        eta = etanp1
        u = unp1
        v = vnp1

    # final output
    print('outputting at t=' + str(t))
    # setup fields dictionary for outputting.
    fields = dict()
    fields["eta"] = eta
    fields["u"] = u
    fields["v"] = v
    fields["N"] = N
    fields["P"] = P
    fields["Z"] = Z
    fields["D"] = D
    outputter.writeFieldsToFiles(fields, step)