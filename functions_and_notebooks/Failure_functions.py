# ========================================================================================================================================================================
# ========================================================================================================================================================================
# ======================================================================================================== Failure Functions =============================================
# ========================================================================================================================================================================
# ========================================================================================================================================================================

import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt

# ========================================================================================================================================================================
# ===================================================================== ThresholdRateAndState_logSampled =================================================================
# ========================================================================================================================================================================

class ThresholdRateAndState_logSampled:
    #def __init__(self,prior_min=[-6.2,-3,-10,-.3],prior_max=[-2.6,0,-0.3,3]): # Note that the priors are sampled in the logarithmic space !! Original !!
    
             #     prior_min=[-4,-4,-6,0],prior_max=[-2,-2,0,5]
      #     prior_min=[-4,-3,-10,0],prior_max=[0,0,0,3]
    
#    def __init__(self,prior_min=[-4,-3,-10,0],prior_max=[0,0,0,6]): # Note that the priors are sampled in the logarithmic space !! Modified by Samson
    def __init__(self,prior_min,prior_max):
        self.prior_min = prior_min
        self.prior_max = prior_max
        self.model     = None
        self.Coulomb   = None
        self.Robs      = None


    def Rates(self,betas): 
      """Returns the seismicity rate predicted by the failure funcion
      for a given set of parameters
      """
#      dCp = 0.5*(self.Coulomb[:,:,1:] + self.Coulomb[:,:,:-1]) # the average coulomb stress change between two epochs
#      dCp = self.Coulomb[:,:,:-1]
      dCp = self.Coulomb[:,:,:]
      betas = np.power(10,betas) # take back the logarithmically sampled parameters
      r, A_sigma0, delta_Sc, t_a = betas[0], betas[1], betas[2], betas[3] #assign the different values to parameters
      K = np.exp( (dCp - delta_Sc) / A_sigma0) # this is the common term in the numerator and denominator
      boolean_threshold = K.__gt__(1)  # this boolean threshold accounts for zero seismicity before \DeltaS_c is reached
      K_threshold = K * boolean_threshold 
      intgr_K = np.cumsum(K_threshold, 2) # integration over time
      Rpred = r * K_threshold / (intgr_K/t_a + 1) #the final seismicity rate
      return Rpred


    def forecast(self,trace,num_chains,num_draws,numsamples=1,compute_logp=False):
        """ Returns a table of height numsamples, each line contains the predicted seismiciy rate over
        the whole timeline from 1 aleatory MCMC iteration
        Returns the betas associated to each iteration
        """
        print ('Forecast Parameters :')
        print (trace)
        print ('chains')
        print (num_chains)
        print ('draws')
        print (num_draws)
        print ('samples')
        print (numsamples)
        with self.model:
            # -- Redefining the coulomb stress model --            
            pm.set_data({'dC':self.Coulomb,'Robs':self.Robs})
#            dCp     = 0.5*(self.Coulomb[:,:,1:] + self.Coulomb[:,:,:-1])
#            dCp = self.Coulomb[:,:,:-1]
            dCp = self.Coulomb[:,:,:]
            indx  = np.zeros((numsamples,2)) #creates the empty table of height = numsamples
            indx[:,0] = np.random.choice(num_chains,numsamples)#indx contains numsamples pairs of chain numbers
            indx[:,1] = np.random.choice(num_draws,numsamples)
            Rpred   = np.zeros((indx.shape[0],dCp.shape[-1]))
            betas   = np.zeros((indx.shape[0],4))

            if compute_logp == True:
                logp    = np.zeros((indx.shape[0]))
                f_logp  = self.model.logp #defined in create_model as logp  = pm.Normal("R", mu=Rpred,sigma=3,observed=robs)

            #loop on the number of samples
            for jj in range(indx.shape[0]):
                    ii = indx[jj,1] #sample number
                    ch = indx[jj,0] #chain number
                    variables   = trace.point(ii,ch) #selecting a random iteration in the MCMC
                    variables = variables
                    
                    r = betas[jj,0] = variables['θ1']  
                    A_sigma0  = betas[jj,1]  = variables['θ2']
                    delta_Sc = betas[jj,2]  = variables['θ3']
                    t_a  = betas[jj,3] = variables['θ4']

                    
                    # go back to linspace from log sampled variables (this is done here again becaise the variables have been assigned). 
                    K = np.exp( (dCp - np.power(10,delta_Sc)) / np.power(10,A_sigma0)) 
                    boolean_threshold = K.__gt__(1) 
                    K_threshold = K * boolean_threshold
                    intgr_K = np.cumsum(K_threshold, 2)
                    #sum over space (axes 0,1 = X,Y) : allows having all the spatial seismicity every sampled time unit
                    Rpred[jj,:] = ( np.power(10,r) * K_threshold / (intgr_K/np.power(10,t_a) + 1) ).sum((0,1))

                    if compute_logp == True:
                      logp[jj]    = f_logp(variables) #variables are taken back to linspace from log sampling in create_model

            if compute_logp == True:
                return Rpred,np.power(10,betas),logp
            else:
                return Rpred,np.power(10,betas)

    def create_model(self):
        '''

        '''
        with pm.Model() as self.model:

            beta1 = pm.Uniform("θ1",self.prior_min[0],self.prior_max[0])
            beta2 = pm.Uniform("θ2",self.prior_min[1],self.prior_max[1])
            beta3 = pm.Uniform("θ3",self.prior_min[2],self.prior_max[2])
            beta4 = pm.Uniform("θ4",self.prior_min[3],self.prior_max[3])
            beta1, beta2, beta3, beta4 = np.power(10,beta1), np.power(10,beta2), np.power(10,beta3), np.power(10,beta4)
            data  = pm.Data('dC',self.Coulomb)
            robs  = pm.Data('Robs',self.Robs)
#            dCp   = 0.5*(data[:,:,1:] + data[:,:,:-1])
#            dCp  = data[:,:,:-1]
            dCp = data[:,:,:]
            K = tt.exp( (dCp - beta3) / beta2)
#            boolean_threshold = tt.gt(K, np.ones(self.Coulomb[:, :, 1:].shape))
            boolean_threshold = tt.gt(K, np.ones(self.Coulomb[:, :, :].shape))
            K_threshold = K * boolean_threshold
            intgr_K = tt.cumsum(K_threshold, 2)
            Rpred = ( beta1 * K_threshold / ((1/beta4)*intgr_K + 1) ).sum((0, 1))
            Rpred_safe = pm.math.maximum(Rpred, 1e-6)
            sigma = pm.math.sqrt(Rpred_safe)
            logp = pm.Normal( "R", mu=Rpred, sigma=sigma, observed=robs)


    def create_model_Poisson(self):
        '''

        '''
        with pm.Model() as self.model:

            beta1 = pm.Uniform("θ1",self.prior_min[0],self.prior_max[0])
            beta2 = pm.Uniform("θ2",self.prior_min[1],self.prior_max[1])
            beta3 = pm.Uniform("θ3",self.prior_min[2],self.prior_max[2])
            beta4 = pm.Uniform("θ4",self.prior_min[3],self.prior_max[3])
            beta1, beta2, beta3, beta4 = np.power(10,beta1), np.power(10,beta2), np.power(10,beta3), np.power(10,beta4)
            data  = pm.Data('dC',self.Coulomb)
            robs  = pm.Data('Robs',self.Robs)
            dCp = data[:,:,:]
            K = tt.exp( (dCp - beta3) / beta2)
#            boolean_threshold = tt.gt(K, np.ones(self.Coulomb[:, :, 1:].shape))
            boolean_threshold = tt.gt(K, np.ones(self.Coulomb[:, :, :].shape))
            K_threshold = K * boolean_threshold
            intgr_K = tt.cumsum(K_threshold, 2)
            Rpred = ( beta1 * K_threshold / ((1/beta4)*intgr_K + 1) ).sum((0, 1))
            logp  = pm.Poisson("R", mu=Rpred,observed=robs)

# ========================================================================================================================================================================
# ================================================================== ClassicRateAndState_logSampled ======================================================================
# ========================================================================================================================================================================
class ClassicRateAndState_logSampled:

    def __init__(self,prior_min,prior_max):
        self.prior_min = prior_min
        self.prior_max = prior_max
        self.model     = None
        self.Coulomb   = None
        self.Robs      = None

#
#    def __init__(self,prior_min=[-8,-4,-1],prior_max=[-2.6,0,3]): # Note that the priors are sampled in the logarithmic space
#        self.prior_min = prior_min
#        self.prior_max = prior_max
#        self.model     = None
#        self.Coulomb   = None
#        self.Robs      = None


    def Rates(self,betas): 
      """Returns the seismicity rate predicted by the failure funcion
      for a given set of parameters
      """
#      dCp = 0.5*(self.Coulomb[:,:,1:] + self.Coulomb[:,:,:-1]) # the average coulomb stress change between two epochs
      dCp = self.Coulomb[:,:,:] # the average coulomb stress change between two epochs
      betas = np.power(10,betas) # take back the logarithmically sampled parameters
      r, A_sigma0, t_a = betas[0], betas[1], betas[2] #assign the different values to parameters
      K = np.exp( (dCp) / A_sigma0) # this is the common term in the numerator and denominator
      boolean_threshold = K.__gt__(1)  # this boolean threshold accounts for zero seismicity before \DeltaS_c is reached
      K_threshold = K * boolean_threshold 
      intgr_K = np.cumsum(K_threshold, 2) # integration over time
      Rpred = r * K_threshold / (intgr_K/t_a + 1) #the final seismicity rate
      return Rpred


    def forecast(self,trace,num_chains,num_draws,numsamples=1,compute_logp=False):
        """ Returns a table of height numsamples, each line contains the predicted seismiciy rate over
        the whole timeline from 1 aleatory MCMC iteration
        Returns the betas associated to each iteration
        """
        print ('Forecast Parameters :')
        print (trace)
        print ('chains')
        print (num_chains)
        print ('draws')
        print (num_draws)
        print ('samples')
        print (numsamples)
        with self.model:
            # -- Redefining the coulomb stress model --            
            pm.set_data({'dC':self.Coulomb,'Robs':self.Robs})
#            dCp     = 0.5*(self.Coulomb[:,:,1:] + self.Coulomb[:,:,:-1])
            dCp = self.Coulomb[:,:,:] # the average coulomb stress change between two epochs
            indx  = np.zeros((numsamples,2)) #creates the empty table of height = numsamples
            indx[:,0] = np.random.choice(num_chains,numsamples)#indx contains numsamples pairs of chain numbers
            indx[:,1] = np.random.choice(num_draws,numsamples)
            Rpred   = np.zeros((indx.shape[0],dCp.shape[-1]))
            betas   = np.zeros((indx.shape[0],3))

            if compute_logp == True:
                logp    = np.zeros((indx.shape[0]))
                f_logp  = self.model.logp #defined in create_model as logp  = pm.Normal("R", mu=Rpred,sigma=3,observed=robs)

            #loop on the number of samples
            for jj in range(indx.shape[0]):
                    ii = indx[jj,1] #sample number
                    ch = indx[jj,0] #chain number
                    variables   = trace.point(ii,ch) #selecting a random iteration in the MCMC
                    variables = variables
                    
                    r = betas[jj,0] = variables['θ1']  
                    A_sigma0  = betas[jj,1]  = variables['θ2']
                    t_a  = betas[jj,2] = variables['θ3']
                    
                    # go back to linspace from log sampled variables (this is done here again becaise the variables have been assigned). 
                    K = np.exp( (dCp) / np.power(10,A_sigma0)) 
                    boolean_threshold = K.__gt__(1) 
                    K_threshold = K * boolean_threshold
                    intgr_K = np.cumsum(K_threshold, 2)
                    #sum over space (axes 0,1 = X,Y) : allows having all the spatial seismicity every sampled time unit
                    Rpred[jj,:] = ( np.power(10,r) * K_threshold / (intgr_K/np.power(10,t_a) + 1) ).sum((0,1))

                    if compute_logp == True:
                      logp[jj]    = f_logp(variables) #variables are taken back to linspace from log sampling in create_model

            if compute_logp == True:
                return Rpred,np.power(10,betas),logp
            else:
                return Rpred,np.power(10,betas)

    def create_model(self):
        '''

        '''
        with pm.Model() as self.model:

            beta1 = pm.Uniform("θ1",self.prior_min[0],self.prior_max[0])
            beta2 = pm.Uniform("θ2",self.prior_min[1],self.prior_max[1])
            beta3 = pm.Uniform("θ3",self.prior_min[2],self.prior_max[2])
            beta1, beta2, beta3 = np.power(10,beta1), np.power(10,beta2), np.power(10,beta3)
            data  = pm.Data('dC',self.Coulomb)
            robs  = pm.Data('Robs',self.Robs)
#            dCp   = 0.5*(data[:,:,1:] + data[:,:,:-1])
            dCp   = data[:,:,:]

            K = tt.exp( (dCp) / beta2)
            boolean_threshold = tt.gt(K, np.ones(self.Coulomb[:, :, :].shape))
            K_threshold = K * boolean_threshold
            intgr_K = tt.cumsum(K_threshold, 2)
            Rpred = ( beta1 * K_threshold / ((1/beta3)*intgr_K + 1) ).sum((0, 1))
            Rpred_safe = pm.math.maximum(Rpred, 1e-6)
            sigma = pm.math.sqrt(Rpred_safe)
            logp  = pm.Normal("R", mu=Rpred,sigma=sigma,observed=robs)



# ========================================================================================================================================================================
# ======================================================================== ExtremeThreshold ==============================================================================
# ========================================================================================================================================================================

class ExtremeThreshold:
    def __init__(self,prior_min=[0.0,0.0,0.0],prior_max=[15.0,30.0,2.0]):
        self.prior_min = prior_min
        self.prior_max = prior_max

        self.model     = None
        self.Coulomb   = None
        self.Robs      = None

    def Rates(self,betas):
        dCp         = 0.5*(self.Coulomb[:,:,1:] + self.Coulomb[:,:,:-1])
        dCm         = (self.Coulomb[:,:,:-1] - self.Coulomb[:,:,1:]) 
        Rpred = -((betas[2]/(500^2))*(np.exp(betas[0] + betas[1]*dCp)*betas[1]*(dCm/1.0)))
        return Rpred

    def FailureFunc(self,dC,betas):
        return (betas[:,2,:]/(500^2))*np.exp(betas[:,0,:] + betas[:,1,:]*dC)

    def forecast(self,trace,num_chains,num_draws,numsamples=1,compute_logp=False):
        with self.model:
            # -- Redefining the coulomb stress model --            
            pm.set_data({'dC':self.Coulomb,'Robs':self.Robs})
            dCp     = 0.5*(self.Coulomb[:,:,1:] + self.Coulomb[:,:,:-1])
            dCm     = (self.Coulomb[:,:,:-1] - self.Coulomb[:,:,1:]) 
            indx  = np.zeros((numsamples,2))
            #For a given number
            indx[:,0] = np.random.choice(num_chains,numsamples) # from num_chains select numsamples random samples
            indx[:,1] = np.random.choice(num_draws,numsamples) # from num_draws select numsamples random samples
            Rpred   = np.zeros((indx.shape[0],dCp.shape[-1]))
            betas   = np.zeros((indx.shape[0],3))
            if compute_logp == True:
                logp    = np.zeros((indx.shape[0]))
                f_logp  = self.model.logp
            for jj in range(indx.shape[0]):
                    ii = indx[jj,1]
                    ch = indx[jj,0]
                    variables   = trace.point(ii,ch)
                    beta1 = betas[jj,0] = variables['θ1']
                    beta2 = betas[jj,1] = variables['θ2']
                    beta3 = betas[jj,2] = variables['θ3']
                    Rpred[jj,:] = -((beta3/(500^2))*(np.exp(beta1 + beta2*dCp)*beta2*(dCm/1.0))).sum((0,1))
                    # the 500^2 factor corrects for the area over which you are calculating the seismicity
                    # the dCm/1 factor corrects for the timestep over which you are calculating the seismicity in Y. if you calculate over months it should be 1/12. 
                    if compute_logp == True:
                        logp[jj]    = f_logp(variables)

            if compute_logp == True:
                return Rpred,betas,logp
            else:
                return Rpred,betas

    def create_model(self):
        '''

        '''
        with pm.Model() as self.model:
            beta1 = pm.Uniform("θ1",self.prior_min[0],self.prior_max[0])
            beta2 = pm.Uniform("θ2",self.prior_min[1],self.prior_max[1])
            beta3 = pm.Uniform("θ3",self.prior_min[2],self.prior_max[2])
            data  = pm.Data('dC',self.Coulomb)
            robs  = pm.Data('Robs',self.Robs)
            dCp   = 0.5*(data[:,:,1:] + data[:,:,:-1])
            dCm   = (data[:,:,:-1] - data[:,:,1:])
            Rpred = -((beta3/(500^2))*(np.exp(beta1 + beta2*dCp)*beta2*(dCm/1.0))).sum((0,1))
            logp  = pm.Normal("R", mu=Rpred,sigma=3,observed=robs)



    def create_model(self):
        '''

        '''
        with pm.Model() as self.model:
            beta1 = pm.Uniform("θ1",self.prior_min[0],self.prior_max[0])
            beta2 = pm.Uniform("θ2",self.prior_min[1],self.prior_max[1])
            beta3 = pm.Uniform("θ3",self.prior_min[2],self.prior_max[2])
            beta4 = pm.Uniform("θ4",self.prior_min[2],self.prior_max[2])
            beta5 = pm.Uniform("θ5",self.prior_min[2],self.prior_max[2])
            
            data  = pm.Data('dC',self.Coulomb)
            robs  = pm.Data('Robs',self.Robs)
            
            dCp   = 0.5*(data[:,:,1:] + data[:,:,:-1])
            dCm   = (data[:,:,:-1] - data[:,:,1:])
            
            Rpred = -((beta3/(500^2))*(np.exp(beta1 + beta2*dCp)*beta2*(dCm/1.0))).sum((0,1))
            logp  = pm.Normal("R", mu=Rpred,sigma=3,observed=robs)

# ========================================================================================================================================================================
# ======================================================================== Gaussian Failure ==============================================================================
# ========================================================================================================================================================================

class GaussianFailure:
    def __init__(self,prior_min=[0.01,0.01,-2.0],prior_max=[0.75,0.75,15]):
        self.prior_min = prior_min
        self.prior_max = prior_max

        self.model     = None
        self.Coulomb   = None
        self.Robs      = None

    def FailureFunc(self,dC,betas):
        #return (betas[:,2,:]/(500^2))*0.5*(1+scipy.special.erf((dC-betas[:,0,:])/(betas[:,1,:]*sqrt(2))))
        return (betas[:,2,:]/(1000^2))*0.5*(1+scipy.special.erf((dC-betas[:,0,:])/(betas[:,1,:]*sqrt(2))))

    def Rates(self,betas):
        dCp         = 0.5*(self.Coulomb[:,:,1:] + self.Coulomb[:,:,:-1])
        dCm         = (self.Coulomb[:,:,:-1] - self.Coulomb[:,:,1:]) 
        Rpred       = (-(np.exp(-0.5*((0.5*dCp - betas[0])/betas[1])**2) * (1/(betas[1]*np.sqrt(2*np.pi))) * np.exp(betas[2]) * ((dCm)/(1.0/12))))
        return Rpred

    def forecast(self,trace,num_chains,num_draws,numsamples=1,compute_logp=False):
        with self.model:
            # -- Redefining the coulomb stress model --            
            pm.set_data({'dC':self.Coulomb,'Robs':self.Robs})
            dCp     = 0.5*(self.Coulomb[:,:,1:] + self.Coulomb[:,:,:-1])
            dCm     = (self.Coulomb[:,:,:-1] - self.Coulomb[:,:,1:]) 
            indx  = np.zeros((numsamples,2))
            indx[:,0] = np.random.choice(num_chains,numsamples)
            indx[:,1] = np.random.choice(num_draws,numsamples)
            Rpred   = np.zeros((indx.shape[0],dCp.shape[-1]))
            betas   = np.zeros((indx.shape[0],3))
            if compute_logp == True:
                logp    = np.zeros((indx.shape[0]))
                f_logp  = self.model.logp
            for jj in range(indx.shape[0]):
                    ii = indx[jj,1]
                    ch = indx[jj,0]
                    variables   = trace.point(ii,ch)
                    beta1 = betas[jj,0] = variables['θ1']
                    beta2 = betas[jj,1] = variables['θ2']
                    beta3 = betas[jj,2] = variables['θ3']
                    Rpred[jj,:] = (-(np.exp(-0.5*((0.5*dCp - beta1)/beta2)**2) * (1/(beta2*np.sqrt(2*np.pi))) * np.exp(beta3) * ((dCm)/(1.0/12)))).sum((0,1))
                    if compute_logp == True:
                        logp[jj]    = f_logp(variables)

            if compute_logp == True:
                return Rpred,betas,logp
            else:
                return Rpred,betas


    def create_model(self):
        '''

        '''
        with pm.Model() as self.model:
            beta1 = pm.Uniform("θ1",self.prior_min[0],self.prior_max[0])
            beta2 = pm.Uniform("θ2",self.prior_min[1],self.prior_max[1])
            beta3 = pm.Uniform("θ3",self.prior_min[2],self.prior_max[2])
            data  = pm.Data('dC',self.Coulomb)
            robs  = pm.Data('Robs',self.Robs)
            dCp   = 0.5*(data[:,:,1:] + data[:,:,:-1])
            dCm   = (data[:,:,:-1] - data[:,:,1:])
            Rpred = (-(np.exp(-0.5*((0.5*dCp - beta1)/beta2)**2) * (1/(beta2*np.sqrt(2*np.pi))) * np.exp(beta3) * ((dCm)/(1.0/12)))).sum((0,1)) # here the /1.0 is dividing by the timestep in years here.
            logp  = pm.Normal("R", mu=Rpred,sigma=3,observed=robs)

