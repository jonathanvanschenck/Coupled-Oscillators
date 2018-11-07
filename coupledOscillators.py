# -*- coding: utf-8 -*-
"""
Created on Nov 11/7/18, updated:
Written by Jonthan Van Schenck

This module is used to fit angle-resolved reflectance data first with a series
of broadened transitions (gaussian or lorentzian or other), and then extract
the centers of those transitions to fit with a coupled oscillator model.


Dependancies:
numpy:          required for array handling
numpy.linalg:   uses eigh to diagonalize coupled oscillator matrix
scipy.optimize: uses least_squares for curve fitting
matplotlib:     uses pyplot for basic plotting


Functions
loadRef:        imports VASE reflectance .dat files (of single polarization) 
vecAnd:         a vectorized implimentation of "and"
Glorz:          Normalized lorentzian function
Gauss:          implimentation of gaussian which takes 2*sig as broadening
                 input, so as to more closely match the FWHM used by Glorz
cOs:            Calculates the E-vecs and E-vals of a coupled oscillator model


Classes
bumpFit:        fits a single reflectance spectrum with nbump transitions
fit:            fits an experimental dispersion (from bumpFit) to a coupled 
                 oscillator model, with nEx+1 oscillators.
"""
from numpy.linalg import eigh
import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt

def loadRef(loc,head):
    """
    This function loads single polarization VASE reflectance .dat files into
    numpy arrays.
    
    
    Inputs
    loc:        string containing path to the .dat file: 'C:/<Path to .dat>/'
    head:       name of .dat file (omit extension)
    
    
    Output
    res:        Numpy array containing reflectance data. formated with first 
                 column holding wavelength. All following columns hold reflectance
                 parsed by angle of incidence:
                  'nm'    '20deg'      '25 deg'
                 [[300,     0.9      ,   0.87     , ...],
                  [310,     0.91     ,   0.89     , ...],
                  [320,     0.9      ,   0.91     , ...],
                  [330,     0.89     ,   0.81     , ...],
                  ...]
    aoi:        Numpy array holding angles of incidence for each column in 
                 res (after the first column).
    """
    raw = np.genfromtxt(loc+head+'.dat',skip_header=4)[:,1:5]
    aoi = np.unique(raw[:,1])
    out = np.transpose([raw[raw[:,1]==i,2] for i in aoi])
    lam = raw[raw[:,1]==aoi[0],0][:,np.newaxis]
    res = np.hstack([lam,out])
    return res, aoi
    
def vecAnd(listArrays):
    """
    This function applies "and" elementwise across a list of numpy arrays. It 
    is used primarially to create plotting/fitting masks by combining boolean
    arrays.
    
    Input
    listArrays:     A list of 1d bool arrays, formatted where each row is holds
                     the elements of a boolean array to be and'ed:
                     [[bool array #1],[bool array #2],[bool array #3]]
                     
    Output
    res:            A 1-d numpy bool array where each element is the total and
                     of all the corresponding elements in "listArrays":
                     [[#1][0] and [#2][0] and ..., [#1][1] and [#2][1] and ..., ...]
    """
    return np.all(np.array(listArrays).T,axis=1)

def Glorz(x,mu,sig):
    """
    This function implements a normalized lorentzian function
    
    Inputs
    x:          Numpy array holding the domain Glorz is to be applied over
    mu:         The center of the distribution
    sig:        The FWHM
    
    Outputs
    res:        Numpy array holding the y values of Glorz
    """
    return sig/((2*np.pi)*((x-mu)**2+(sig/2)**2))

def Gauss(x,mu,doubsig):
    """
    This function implements a normalized gaussian function, which has broadening
    close to the FWHM, rather than the standard deviation
    
    Inputs
    x:          Numpy array holding the domain Gauss is to be applied over
    mu:         The center of the distribution
    doubsig:    2*standardDeviation of the distribtuion (notice doubsig~FWHM)
    
    Outputs
    res:        Numpy array holding the y values of Gauss
    """
    sig = doubsig/2
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sig**2))
    
def cOs(Ex,Ec,V,nEx=1):
    """
    This function calculates the E-vals and E-vecs of a coupled oscillator model
    
    Inputs
    Ex:         a float representing the bare exciton energy (or array holding 
                 each exciton energy if more than 1)
    Ec:         a float representing the bare photon energy
    V:          a float representing the coupling between Ex and Ec (or array
                 if more than 1 exciton) the length of V MUST match the length
                 of Ex
    nEx:        an integer specifying the number of excitons. nEx MUST match
                 the length of Ex.
                 
    Outputs
    (eval, evec):Tuple holding the eigenvalues and eigenvectors of the system
    eval:       Numpy 1d array holding The eigenvalues (organized smallest to 
                 largest) of the system
    evec:       Numpy 2d array holding the eigenvectors of the system. Vectors
                 are the columns of the 2d array and are organized in the same
                 order as the "eval" 1d array.
    """
    mat = np.zeros((nEx+1,nEx+1))
    if nEx == 1:
        Vp = np.array([V])
        Exp = np.array([Ex])
    else:
        Vp = np.array(V)
        Exp = np.array(Ex)
    mat[0,0] = Ec
    for i in range(nEx):
        mat[i+1,i+1] = Exp[i]
        mat[i+1,0] = Vp[i]
        mat[0,i+1] = Vp[i]    
    return eigh(mat)
    
class bumpFit:
    """
    This module fits reflectance data with a series of broadened transitions,
    fitFun = S + a1*fun(ev,E1,G1) + a2*fun(ev,E2,G2) + ... (either lorentzian 
    or gaussian, or other).
    
    Inputs Required
    l:          1d Numpy array holding the wavelengths of reflectance (in nm)
    ref:        1d Numpy array holding the reflectance measured as each wavlength
    nbump:      Integer indicating the number of transitions used during fitting
    fun:        A function "fun(x,mu,sig)" for the broadened transition. x is 
                 a numpy array holding the domain, mu is the center of the
                 distribution, and sig is the broadening (usually FWHM).
    
    Attributes:
    l:          1d Numpy array holding the wavelengths of reflectance (in nm)
    ref:        1d Numpy array holding the reflectance measured as each wavlength
    nbump:      Integer indicating the number of transitions used during fitting
    fun:        A function "fun(x,mu,sig)" for the broadened transition. x is 
                 a numpy array holding the domain, mu is the center of the
                 distribution, and sig is the broadening (usually FWHM).
    ev:         1d Numpy array holding the photon energy of reflectance (in eV)         
    paramNames: 1d Numpy array holding the names of each parameter:
                 S:  is the baseline reflectance level, usually ~0.8-1.0
                 ai: is the area of the ith broadened transition
                 Ei: is the center energy of the ith broadened transition
                 Gi: is the broadening of the ith broadened transition
    iparam:     1d numpy array holding the initial guess for paramter values
                 (must be specified using .initalizeFitParams BEFORE the fit
                  can be performed). Structure:
                  [S,a1,E1,G1,a2,E2,G2,...]
    param:      1d numpy array holding the resulting parameter values after 
                 fitting. Structure:
                  [S,a1,E1,G1,a2,E2,G2,...]
    which:      1d bool array holding specifying which of the parameters will
                 be allowed to varrying during fitting. Default is to allow all
                 parameters to varry. Can be modified using .freezeFitParams.
                 Structure:
                  [S?,a1?,E1?,G1?,a2?,E2?,G2?,...]
    bound:      2d numpy array holding the paramater bounds to be used during 
                 fitting. If parameters have been frozen by using .freezeFitParam
                 method, then bound will only contain bounds for the parameters
                 which are used during fitting. i.e. bound.shape[1]=nf. 
                 Bound[0] is the lower bound and bound[1] is the upper
                 bound. Note, iparam[i] MUST be in range (bound[0][i],bound[1][i])
                 Structure when no parameters are frozen:
                  [[S_,a1_,E1_,G1_,a2_,E2_,G2_,...],
                   [S^,a1^,E1^,G1^,a2^,E2^,G2^,...]]
    nf:         Value holding the number of parameters allowed to varry
    fitMask:    1d bool array specifying which reflectance data points to be 
                 used during fitting. Default is to use all reflectance data.
                 Can be modified using .createFitRegion
    plotMask:   1d bool array specifying which reflectance data points to be 
                 used during plotting. Default is to use all reflectance data.
                 Can be modified using .createPlotRegion
                 
    
    Best Practice for Use:
          1)  Call bumpFit class and provide l, ref, nubmp and Glorz
       / 2a)  Specify fit region (.createFitRegion)
    opt| 2b)  Specify plot region (.createPlotRegion)
       \ 2c)  Freeze parameters NOT used during fitting (.freezeFitParams)
          3)  Provide inital guess for parameter values (.initalizeFitParams)
          4)  Set bounds on free fit parameters (.createFitParamBounds)
          5)  Perform Fit (.performFit)
     opt/ 6)  Plot resuts (.plot)
     opt\ 7)  Print fit results (.printParam)
        
    
    """
    def __init__(self,l,ref,nbump=2,fun=Glorz):
        """
        This method sets up the fitting proceedure
        Input
        l:          1d Numpy array holding the wavelengths of reflectance (in nm)
        ref:        1d Numpy array holding the reflectance measured as each wavlength
        nbump:      Integer indicating the number of transitions used during fitting
        fun:        A function "fun(x,mu,sig)" for the broadened transition. x is 
                     a numpy array holding the domain, mu is the center of the
                     distribution, and sig is the broadening (usually FWHM).
        """
        self.l = l
        self.ref = ref
        self.ev = 1240/self.l        
        self.nbump = nbump
        test0 = lambda x: ['a'+x,'E'+x,'G'+x]
        self.paramNames = np.hstack([test0(i) for i in np.vectorize(str)(np.arange(1,nbump+1))])
        self.paramNames = np.hstack([['S'],self.paramNames])
        self.iparam = np.zeros(1+nbump*3)
        self.param = np.copy(self.iparam)
        self.which = np.full(1+nbump*3,True,dtype='bool')
        self.nf = np.sum(np.ones(1+3*self.nbump)[self.which])
        self.fitMask = np.full(len(self.l),True,'bool')
        self.plotMask = np.full(len(self.l),True,'bool')
        self.fun = fun
        self.bound = np.array((2,len(self.paramNames)))
    
    def createFitRegion(self,mini,maxi,eV=True):
        """
        Function creates a boolean mask for data to be use during fitting to
        select which data points to fit by. Allows one to select a range of
        either photon energies, or wavelengths inside which to fit.
        
        Input
        mini:       Float: left bound on fit range
        maxi:       Float: right bound on fit range
        ev:         Boolean: when true, fit range is over photon energy (in eV)
                     when false, fit range is selected over wavelength. Note
                     that fits are always assume energy domain, so wavelengths 
                     will later be converted into energies.
        """
        if eV:
            self.fitMask = vecAnd([self.ev<maxi,self.ev>mini])
        else:
            self.fitMask = vecAnd([(1240/self.ev)<maxi,(1240/self.ev)>mini])
            
    def createPlotRegion(self,mini,maxi,eV=True):
        """
        Function creates a boolean mask for data to be use during plotting to
        select which data points to plot by. Allows one to select a range of
        either photon energies, or wavelengths inside which to fit.
        
        Input
        mini:       Float: left bound on fit range
        maxi:       Float: right bound on fit range
        ev:         Boolean: when true, plot range is over photon energy (in eV)
                     when false, plot range is selected over wavelength. Note
                     that plots are always assume energy domain, so wavelengths
                     will later be converted into energies.
        """
        if eV:
            self.plotMask = vecAnd([self.ev<maxi,self.ev>mini])
        else:
            self.plotMask = vecAnd([(1240/self.ev)<maxi,(1240/self.ev)>mini])
            
    def freezeFitParams(self,which):
        """
        Function allows user to freeze particular parameters during fitting. 
        By specifying boolean "False" for a parameter, it is not allowed to vary
        during fitting. Structure: [S?,a1?,E1?,G1?,a2?,E2?,G2?,...]
        
        Input
        which:      1d Boolean list/array. Array MUST be the same length as 
                     iparam/param/paramNames. If which[i]==True, than param[i]
                     is allowed to vary during fitting. If which[i]==False, 
                     than param[i] is frozen during fitting.
        """
        self.which = np.array(which,dtype='bool')
        self.nf = np.sum(np.ones(1+3*self.nbump)[self.which])
    
    def initalizeFitParams(self,iparam):
        """
        Function sets the initial guess for parameter values
        
        Input
        iparam:     1d array/list which holds the initial guess for each
                     parameter value. The length MUST be 1+nbump*3. Structure:
                     [S,a1,E1,G1,a2,E2,G2,...]
        """
        if len(iparam)!=1+self.nbump*3:
            print('Incorrect number of parameters, try again')
        else:
            self.iparam = np.array(iparam)
            self.param = np.copy(self.iparam)
    
    def createFitParamBounds(self,bound):
        """
        Function sets the bounds for parameter values during fitting
        
        Input
        bound:      2d numpy array holding the paramater bounds to be used during 
                     fitting. If parameters have been frozen by using .freezeFitParam
                     method, then bound will only contain bounds for the parameters
                     which are used during fitting. i.e. bound.shape[1]=nf. 
                     Bound[0] is the lower bound and bound[1] is the upper
                     bound. Note, iparam[i] MUST be in range (bound[0][i],bound[1][i])
                     Structure:
                         [[S_,a1_,E1_,G1_,a2_,E2_,G2_,...],
                          [S^,a1^,E1^,G1^,a2^,E2^,G2^,...]]
        """
        self.bound = bound
    
    def fitFun(self,par):
        """
        Function which is being fit to data. A series of broadened transitions.
        fitFun = S + a1*fun(ev,E1,G1) + a2*fun(ev,E2,G2) + ...
        
        Input
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting
        
        Output
        res:        1d numpy array holding the simulated reflectance at each
                     provided photon energy.
        """
        p = np.copy(self.param)
        p[self.which] = np.array(par)
        res = p[0]*np.ones(len(self.l))
        for i in range(self.nbump):
            j = int((i)*3)+1
            res += p[j+0]*self.fun(self.ev,p[j+1],p[j+2])
            #print(j,j+1,j+2)
        return res
        
    def fitFunDifference(self,par):
        """
        Function gives the error between fit and data. Used by 
        scipy.optimize.least_squares to minimize the SSE.
        
        Input:
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting
                     
        Output
        res:        1d numpy array holding the error between fit and data
        """
        return (self.fitFun(par)[self.fitMask]-self.ref[self.fitMask])
    
    def plot(self,plotName=''):
        """
        Function gives a plot of the data and fit. Must be called AFTER 
        .initializeFitParams, but can be called before .performFit.
        
        Input
        plotName:   String which titles the plot.
        """
        plt.figure()
        data, = plt.plot(self.ev[self.plotMask],self.ref[self.plotMask],'o',label='Data')
#        fit, = plt.plot(self.ev[self.plotMask],
#                       (self.fitFun(self.param[self.which]))[self.plotMask],label='fit ex')
        fit, = plt.plot(self.ev[self.fitMask],
                       (self.fitFun(self.param[self.which]))[self.fitMask],label='fit',color='cyan')
        for i in range(self.nbump):
            j = int((i)*3)+1
            plt.plot(self.ev[self.fitMask],(self.param[0]+self.param[j+0]*self.fun(self.ev,self.param[j+1],self.param[j+2]))[self.fitMask],label='fit'+str(i+1))
        plt.title(plotName)
        plt.legend(loc='lower right')
        plt.show()
        
    def performFit(self,xtol=3e-16,ftol=1e-10,num=6):
        """
        Function modifies param[which] so as to minimize the SSE using
        scipy.optimize.least_squares.
        
        Input
        xtol:       See least_squares documentation
        ftol:       See least_squares documentation
        num:        Integer holding the number of parameters to be printed on
                     each line
        
        Output
        res:        Prints out "Start" iparam[which], "End" param[which] and 
                     "Shift" (param-iparam)[which] as a percentage of upper and
                     lower bounds. This is used to see if any parameters have 
                     "hit" the edges of their range during fitting. This can be
                     seen by as "End" being either 0.0 or 1.0. "Start" can be 
                     used to see if the bounds are too loose, or too strict.
                     And "Shift" gives a sense for how good the initial guess
                     was.
        """
        self.fit = least_squares(self.fitFunDifference,self.iparam[self.which],
                                 verbose=1,bounds=self.bound,xtol=xtol,ftol=ftol)
        if self.fit.success:
            self.param = np.copy(self.iparam)
            self.param[self.which] = np.copy(self.fit.x)
        else:
            print('Fit Falue, see: self.fit.message')
            self.param = np.copy(self.iparam)
        start = (self.iparam[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        end = (self.param[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        difference = (self.param[self.which]-self.iparam[self.which])/(np.array(self.bound[1])-np.array(self.bound[0]))
        st = lambda x: '{0:6.3f}'.format(x)
        st2 = lambda x: "{0:>6}".format(x)        
        if len(self.paramNames[self.which])%num==0:
            setp = np.arange(len(self.paramNames[self.which])//num)
        else:
            setp = np.arange((len(self.paramNames[self.which])//num)+1)
        for i in setp:
            print(np.hstack([np.array([[' Name'],['Start'],['  End'],['Shift']]),
                             np.vstack([np.vectorize(st2)(self.paramNames[self.which][(num*i):(num*(i+1))]),
                                       np.vectorize(st)(start[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(end[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(difference[(num*i):(num*(i+1))])])
                            ]))     
#        print(np.hstack([np.array([[' Name'],['Start'],['  End'],['Shift']]),
#                         np.vstack([np.vectorize(st2)(self.paramNames[self.which]),
#                                   np.vectorize(st)(start),
#                                   np.vectorize(st)(end),
#                                   np.vectorize(st)(difference)])
#                        ]))
                                
    def printParam(self,num=6):
        """
        Function prints out the parameter values and names.
        
        Input
        num:        Integer specifying the number of parameters to print onto
                     each line
        """
        st = lambda x: "{0:6.3f}".format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames)%num==0:
            setp = np.arange(len(self.paramNames)//num)
        else:
            setp = np.arange((len(self.paramNames)//num)+1)
        for i in setp:
            print(np.hstack([[[' Name'],['Value']],
                             np.vstack([
                                        np.vectorize(st2)(self.paramNames[(num*i):(num*(i+1))]),
                                        np.vectorize(st)(self.param[(num*i):(num*(i+1))])
                                        ])
                            ]))

class fit:
    """
    This module fits dispersion of reflectance data to a coupled oscillator
    model:
    [[ Ec , V1 , V2 , V3 ,...],
     [ V1 , Ex1, 0.0, 0.0,...],
     [ V2 , 0.0, Ex2, 0.0,...],
     [ V3 , 0.0, 0.0, Ex3,...],
     ...
     ]
    Where Ec=Ec0/np.sqrt(1-(np.sin(np.pi*aoi/180)/neff)**2)
    Where Ec= hc/(2*L*neff), with L=cavity thickness
    
    Inputs Required
    aoi:        1d Numpy array holding the angles of incidence (in degrees)
    pts:        list of Numpy arrays holding transition energies at each aoi. 
                 Structure: [[E@aoi[0],E@aoi[1],...]#Lowest energy dispersion,
                             [E@aoi[0],E@aoi[1],...]#Next energy dispersion,
                             [E@aoi[0],E@aoi[1],...]#Next energy dispersion,
                             ...,
                             [E@aoi[0],E@aoi[1],...]#Highest energy dispersion]
    nEx:        Integer indicating the number excitons used in the coupled
                 oscillator model. If (nEx+1)>pts.shape[0], i.e. if the model
                 expects more dispersions lines than the input data provides,
                 than the extra excitons will sit above the experimental data
                 and will not be directly fit to any data points.
    
    Attributes:
    aoi:        1d Numpy array holding the angles of incidence (in degrees)
    pts:        list of Numpy arrays holding transition energies at each aoi. 
                 Structure: [[E@aoi[0],E@aoi[1],...]#Lowest energy dispersion,
                             [E@aoi[0],E@aoi[1],...]#2nd energy dispersion,
                             [E@aoi[0],E@aoi[1],...]#3rd energy dispersion,
                             ...,
                             [E@aoi[0],E@aoi[1],...]#(nE)th energy dispersion]
    nE:         Integer holding the number of dispersion curves in experimental
                 data.
    nEx:        Integer indicating the number excitons used in the coupled
                 oscillator model. If (nEx+1)>En, i.e. if the model expects 
                 more dispersions lines than the input data provides,than the 
                 extra excitons will sit above the experimental data and will 
                 not be directly fit to any data points.       
    paramNames: 1d Numpy array holding the names of each parameter:
                 Ec0:  The empty cavity resonance at normal incidence
                 neff: The effective index of refraction inside the cavity
                 Exi:  The bare exciton energy of the ith exciton
                 Vi:   The Photon-Exciton coupling for the ith exciton. The
                         rabi splitting is 2*Vi
    iparam:     1d numpy array holding the initial guess for paramter values
                 (must be specified using .initalizeFitParams BEFORE the fit
                  can be performed). Structure:
                  [Ec0,neff,Ex1,V1,Ex2,V2,...]
    param:      1d numpy array holding the resulting parameter values after 
                 fitting. Structure:
                  [Ec0,neff,Ex1,V1,Ex2,V2,...]
    which:      1d bool array holding specifying which of the parameters will
                 be allowed to varrying during fitting. Default is to allow all
                 parameters to varry. Can be modified using .freezeFitParams.
                 Structure:
                  [Ec0?,neff?,Ex1?,V1?,Ex2?,V2?,...]
    bound:      2d numpy array holding the paramater bounds to be used during 
                 fitting. If parameters have been frozen by using .freezeFitParam
                 method, then bound will only contain bounds for the parameters
                 which are used during fitting. i.e. bound.shape[1]=nf. 
                 Bound[0] is the lower bound and bound[1] is the upper
                 bound. Note, iparam[i] MUST be in range (bound[0][i],bound[1][i])
                 Structure which no params are frozen:
                  [[Ec0_,neff_,Ex1_,V1_,Ex2_,V2_,...],
                   [Ec0^,neff^,Ex1^,V1^,Ex2^,V2^,...]]
    nf:         Value holding the number of parameters allowed to varry
    fitMask:    1d bool array specifying which reflectance data points to be 
                 used during fitting. Default is to use all dispersion data.
                 Can be modified using .createFitRegion
                 
    
    Best Practice for Use:
          1)  Call fit class and provide aoi, pts and nEx
    opt/ 2a)  Specify fit region (.createFitRegion)
    opt\ 2b)  Freeze parameters NOT used during fitting (.freezeFitParams)
          3)  Provide inital guess for parameter values (.initalizeFitParams)
          4)  Set bounds on free fit parameters (.createFitParamBounds)
          5)  Perform Fit (.performFit)
     opt/ 6)  Plot resuts (.plot)
     opt\ 7)  Print fit results (.printParam)
    """
    def __init__(self,aoi,pts,nEx):
        """
        aoi:        1d Numpy array holding the angles of incidence (in degrees)
        pts:        list of Numpy arrays holding transition energies at each aoi. 
                     Structure: [[E@aoi[0],E@aoi[1],...]#Lowest energy dispersion,
                                 [E@aoi[0],E@aoi[1],...]#Next energy dispersion,
                                 [E@aoi[0],E@aoi[1],...]#Next energy dispersion,
                                   ...,
                                 [E@aoi[0],E@aoi[1],...]#Highest energy dispersion]
        nEx:        Integer indicating the number excitons used in the coupled
                     oscillator model. If (nEx+1)>pts.shape[0], i.e. if the model
                     expects more dispersions lines than the input data provides,
                     than the extra excitons will sit above the experimental data
                     and will not be directly fit to any data points.
        """
        self.aoi = aoi
        self.pts = pts     
        self.nE = pts.shape[0]
        self.nEx = nEx
        test0 = lambda x: ['Ex'+x,'V'+x]
        self.paramNames = np.hstack([test0(i) for i in np.vectorize(str)(np.arange(1,self.nEx+1))])
        self.paramNames = np.hstack([['Ec0','neff'],self.paramNames])
        self.iparam = np.zeros(len(self.paramNames))
        self.param = np.copy(self.iparam)
        self.which = np.full(len(self.paramNames),True,dtype='bool')
        self.nf = np.sum(np.ones(len(self.paramNames))[self.which])
        self.fitMask = np.full(len(self.aoi),True,'bool')
        self.bound = np.array((2,len(self.paramNames)))
    
    def createFitRegion(self,mini,maxi):
        """
        Function creates a boolean mask for data to be use during fitting to
        select which aoi points to fit by.
        
        Input
        mini:       Float: left bound on aoi range
        maxi:       Float: right bound on aoi range
        """
        self.fitMask = vecAnd([self.aoi<maxi,self.aoi>mini])
                    
    def freezeFitParams(self,which):
        """
        Function allows user to freeze particular parameters during fitting. 
        By specifying boolean "False" for a parameter, it is not allowed to vary
        during fitting. Structure: [Ec0?,neff?,Ex1?,V1?,Ex2?,V2?,...]
        
        Input
        which:      1d Boolean list/array. Array MUST be the same length as 
                     iparam/param/paramNames. If which[i]==True, than param[i]
                     is allowed to vary during fitting. If which[i]==False, 
                     than param[i] is frozen during fitting.
        """
        self.which = np.array(which,dtype='bool')
        self.nf = np.sum(np.ones(len(self.paramNames))[self.which])
    
    def initalizeFitParams(self,iparam):
        """
        Function sets the initial guess for parameter values
        
        Input
        iparam:     1d array/list which holds the initial guess for each
                     parameter value. The length MUST be 2+2*nEx. Structure:
                     [Ec0,neff,Ex1,V1,Ex2,V2,...]
        """
        self.iparam = np.array(iparam)
        self.param = np.copy(self.iparam)
    
    def createFitParamBounds(self,bound):
        """
        Function sets the bounds for parameter values during fitting
        
        Input
        bound:      2d numpy array holding the paramater bounds to be used during 
                     fitting. If parameters have been frozen by using .freezeFitParam
                     method, then bound will only contain bounds for the parameters
                     which are used during fitting. i.e. bound.shape[1]=nf. 
                     Bound[0] is the lower bound and bound[1] is the upper
                     bound. Note, iparam[i] MUST be in range (bound[0][i],bound[1][i])
                     Structure when not parameters are frozen:
                         [[Ec0_,neff_,Ex1_,V1_,Ex2_,V2_,...],
                          [Ec0^,neff^,Ex1^,V1^,Ex2^,V2^,...]]
        """
        self.bound = bound
    
    def fitFun(self,par):
        """
        Function to fit to data. Takes in the free parameters and simulates
        a coupled oscillator system with these parameters at all of the 
        angles of incidence provided.
        
        Input
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting
        
        Output
        res:        2d Numpy array holding the simulated dispersion. Structure: 
                     [[E@aoi[0],E@aoi[1],...]#Lowest energy dispersion,
                      [E@aoi[0],E@aoi[1],...]#2nd energy dispersion,
                      [E@aoi[0],E@aoi[1],...]#3rd energy dispersion,
                          ...,
                      [E@aoi[0],E@aoi[1],...]#(nEx+1)th energy dispersion]
        """
        p = np.copy(self.param)
        p[self.which] = np.array(par)
        Ecf = lambda th: p[0]/np.sqrt(1-(np.sin(np.pi*th/180)/p[1])**2)
        Ecl = np.vectorize(Ecf)(self.aoi)
        index = np.vectorize(int)(np.arange(0,self.nEx))
        res = np.array([cOs(p[2*index+2],Ec,p[2*index+3],nEx=self.nEx)[0] for Ec in Ecl])
        return np.transpose(res)
        
    def fitFunDifference(self,par):
        """
        Function gives the error between fit and data. Used by 
        scipy.optimize.least_squares to minimize the SSE.
        
        Input:
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting
                     
        Output
        res:        1d numpy array holding the error between fit and data.
                     Note that the result of .fitFun and pts are 2d arrays, 
                     which have been collasped into a 1d array by np.hstack
        """
        return np.hstack(self.fitFun(par)[0:(self.nE)]-self.pts)
    
    def plot(self,plotName=''):
        """
        Function gives a plot of the data and fit. Must be called AFTER 
        .initializeFitParams, but can be called before .performFit.
        
        Input
        plotName:   String which titles the plot.
        """
        plt.figure()
        data = [plt.plot(self.aoi,i,'o',label='Data') for i in self.pts]
        fit = [plt.plot(self.aoi,
                        i,label='fit',color='cyan') for i in self.fitFun(self.param[self.which])]
        plt.title(plotName)
        plt.legend(loc='lower right')
        #plt.show()
        
    def performFit(self,xtol=3e-16,ftol=1e-10,num=6):
        """
        Function modifies param[which] so as to minimize the SSE using
        scipy.optimize.least_squares.
        
        Input
        xtol:       See least_squares documentation
        ftol:       See least_squares documentation
        num:        Integer holding the number of parameters to be printed on
                     each line
                     
        Output
        res:        Prints out "Start" iparam[which], "End" param[which] and 
                     "Shift" (param-iparam)[which] as a percentage of upper and
                     lower bounds. This is used to see if any parameters have 
                     "hit" the edges of their range during fitting. This can be
                     seen by as "End" being either 0.0 or 1.0. "Start" can be 
                     used to see if the bounds are too loose, or too strict.
                     And "Shift" gives a sense for how good the initial guess
                     was.
        """
        self.fit = least_squares(self.fitFunDifference,self.iparam[self.which],
                                 verbose=1,bounds=self.bound,xtol=xtol,ftol=ftol)
        if self.fit.success:
            self.param = np.copy(self.iparam)
            self.param[self.which] = np.copy(self.fit.x)
        else:
            print('Fit Falue, see: self.fit.message')
            self.param = np.copy(self.iparam)
        start = (self.iparam[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        end = (self.param[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        difference = (self.param[self.which]-self.iparam[self.which])/(np.array(self.bound[1])-np.array(self.bound[0]))
        st = lambda x: '{0:6.3f}'.format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames[self.which])%num==0:
            setp = np.arange(len(self.paramNames[self.which])//num)
        else:
            setp = np.arange((len(self.paramNames[self.which])//num)+1)
        for i in setp:
            print(np.hstack([np.array([[' Name'],['Start'],['  End'],['Shift']]),
                             np.vstack([np.vectorize(st2)(self.paramNames[self.which][(num*i):(num*(i+1))]),
                                       np.vectorize(st)(start[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(end[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(difference[(num*i):(num*(i+1))])])
                            ]))     
#        print(np.hstack([np.array([[' Name'],['Start'],['  End'],['Shift']]),
#                         np.vstack([np.vectorize(st2)(self.paramNames[self.which]),
#                                   np.vectorize(st)(start),
#                                   np.vectorize(st)(end),
#                                   np.vectorize(st)(difference)])
#                        ]))
                                
    def printParam(self,num=6):
        """
        Function prints out the parameter values and names.
        
        Input
        num:        Integer specifying the number of parameters to print onto
                     each line
        """
        st = lambda x: "{0:6.3f}".format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames)%num==0:
            setp = np.arange(len(self.paramNames)//num)
        else:
            setp = np.arange((len(self.paramNames)//num)+1)
        for i in setp:
            print(np.hstack([[[' Name'],['Value']],
                             np.vstack([
                                        np.vectorize(st2)(self.paramNames[(num*i):(num*(i+1))]),
                                        np.vectorize(st)(self.param[(num*i):(num*(i+1))])
                                        ])
                            ]))