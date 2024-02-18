import numpy as np
from . import utils
import tqdm

average_function_by_i = [None] * 10
average_function_by_i[1] = lambda X: (X[...,0])/1
average_function_by_i[2] = lambda X: (X[...,0]+X[...,1])/2
average_function_by_i[3] = lambda X: (X[...,0]+X[...,1]+X[...,2])/3
average_function_by_i[4] = lambda X: (X[...,0]+X[...,1]+X[...,2]+X[...,3])/4
average_function_by_i[5] = lambda X: (X[...,0]+X[...,1]+X[...,2]+X[...,3]+X[...,4])/5
average_function_by_i[6] = lambda X: (X[...,0]+X[...,1]+X[...,2]+X[...,3]+X[...,4]+X[...,5])/6
average_function_by_i[7] = lambda X: (X[...,0]+X[...,1]+X[...,2]+X[...,3]+X[...,4]+X[...,5]+X[...,6])/7
average_function_by_i[8] = lambda X: (X[...,0]+X[...,1]+X[...,2]+X[...,3]+X[...,4]+X[...,5]+X[...,6]+X[...,7])/8
average_function_by_i[9] = lambda X: (X[...,0]+X[...,1]+X[...,2]+X[...,3]+X[...,4]+X[...,5]+X[...,6]+X[...,7]+X[...,8])/9

def average_over_last_axis(X):
    if X.shape[-1] < len(average_function_by_i):
        return average_function_by_i[X.shape[-1] ](X)
    else:
        return np.mean(X, axis = -1)

MoyenneSousElemPiezo = average_over_last_axis()

def mypfield(x : np.ndarray,y : np.ndarray,z : np.ndarray,DelaiEmissionElemPiezo  : np.ndarray,param):
    FrequencyStep = 1 
    RP = 0 # RP = Radiation Pattern
    # RP = np.zero((-1,param.Nelements, delaysTX.shape[0] )) 
    RadiusOfCurvature = param.radius
    c = 1540 # % default value of sound speed
    OriginalShape = x.shape

    # FREQUENCY SAMPLES
    # param.fc = central frequency (Hz)
    Nf = int( 2* np.ceil( param.fc )+1) # number of frequency samples
    Nf = 10000
    f = np.linspace(0,2*param.fc,Nf) # frequency samples
    nSampling = len(f)

    # Position des elements de la sonde
    # THe = angle of the normal to element with respect to the z-axis
    ElementWidth = param.width
    NumberOfElements = param.Nelements
    PositionsCentresElemPiezo_x, PositionsCentresElemPiezo_z, AngleNormalElemPiezo, _ = param.getElementPositions()
    # AngleNormalElemPiezo = 0
    PositionsCentresElemPiezo_x = PositionsCentresElemPiezo_x.reshape((1,NumberOfElements,1))
    PositionsCentresElemPiezo_z = PositionsCentresElemPiezo_z.reshape((1,NumberOfElements,1))

    ResolutionSpatiale = c/(param.fc*(1+param.bandwidth/200))

    NbSousElemPiezo = int(np.ceil(ElementWidth/ResolutionSpatiale))
    #NbSousElemPiezo=1
    SizeSousElemPiezo = ElementWidth/NbSousElemPiezo

    PositionsCentresSousElemPiezo = -ElementWidth/2 + SizeSousElemPiezo/2 + np.arange(NbSousElemPiezo)*SizeSousElemPiezo
    PositionsCentresSousElemPiezo = PositionsCentresSousElemPiezo.reshape((1,1,NbSousElemPiezo)) # une valeur x,z pour chaque SousElemPiezo

    PositionsCentresSousElemPiezo_x = PositionsCentresSousElemPiezo*np.cos(AngleNormalElemPiezo)[:,:,np.newaxis]
    PositionsCentresSousElemPiezo_z = PositionsCentresSousElemPiezo*np.sin(-AngleNormalElemPiezo)[:,:,np.newaxis]

   
    #%-- Coordinates of the points where pressure is needed
    x = x.reshape((-1,1,1)).astype(np.float32)
    y = y.reshape((-1,1,1)).astype(np.float32)
    z = z.reshape((-1,1,1)).astype(np.float32)


    FreqSweep = param.TXfreqsweep
    #%-- FREQUENCY SPECTRUM of the transmitted pulse
    pulseSpectrum = param.getPulseSpectrumFunction(FreqSweep)
    #%-- FREQUENCY RESPONSE of the ensemble PZT + probe
    probeSpectrum = param.getProbeFunction()
    pulseSPECT = pulseSpectrum(2*np.pi*f) # pulse spectrum
    probeSPECT = probeSpectrum(2*np.pi*f) # probe 

  
    # valeurs de RP pour chaque point de la grille pour chaque sous element piezo de chaque elem de la sonde 
    dxi = x.reshape((-1,1,1)) - PositionsCentresSousElemPiezo_x - PositionsCentresElemPiezo_x
    dzi = z.reshape((-1,1,1)) - PositionsCentresSousElemPiezo_z - PositionsCentresElemPiezo_z
    dyi = y.reshape((-1,1,1))
    r2 = dxi**2 + dzi**2 + dyi**2 # pythagore 3d

    # Pas de limite inferieure de resolution spatiale
    # Tout peut etre aussi petit que necessaire, ça ne pose pas de probleme
    # smallD2 = ResolutionSpatiale**2
    # d2[d2<smallD2] = smallD2

    r = np.sqrt(r2).astype(np.float32) # distance between the segment centroid and the point of interest
    # On considere que la frequemce est independente de la direction du signal emis/recu. Ne depend uniquement de la distance (r)
    # Th = np.arcsin(dxi/r)-AngleNormalElemPiezo.reshape((1,-1,1))
    del r2

    # # %-- EXPONENTIAL arrays of size [numel(x) NumberOfElements M]
    wavenumber = 2*np.pi*f[0]/c # % wavenumber
    kwa = 0 # % attenuation-based wavenumber; no attenuation -> 0
    # EXP = np.exp(-kwa*r + 1j*np.mod(kw*r,2*np.pi)).astype(np.complex64) #; % faster than exp(-kwa*r+1j*kw*r)
    EXP = np.exp(-kwa*r+1j*wavenumber*r).astype(np.complex64)

    dkw = 2*np.pi*FrequencyStep/c
    dkwa = 0
    delta_EXP = np.exp(-dkwa*r + 1j*dkw*r).astype(np.complex64)

    del r
    
    for k  in tqdm.tqdm(range(nSampling)):

        wavenumber_k = 2*np.pi*f[k]/c #; % wavenumber

        EXP = EXP*delta_EXP

        ValeurElemPiezo = MoyenneSousElemPiezo(EXP); #% summation over the M small segments
        
        DelaiEmissionElemPiezo = DelaiEmissionElemPiezo.astype(np.float32)

        APOD = 1 # pas d'apodization -> APOD=1
        DELAPOD = APOD * np.sum(np.exp(1j*wavenumber_k*c*DelaiEmissionElemPiezo), 0).reshape((-1, 1)) # force (NumberofElements,1)

        RPk = np.matmul(ValeurElemPiezo, DELAPOD) #somme des radiation pattern de chaque élément
        
        RPk = pulseSPECT[k]*probeSPECT[k]*RPk #%- include spectrum responses:

        RP += np.abs(RPk)**2


    RP = np.sqrt(RP); #% acoustic intensity, RPk est le radiation pattern de chaque élément

    CorrectingFactor = FrequencyStep
    RP = RP*CorrectingFactor
    return RP.reshape(OriginalShape)