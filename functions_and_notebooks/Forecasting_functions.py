# ========================================================================================================================================================================
# ========================================================================================================================================================================
# ======================================================================================================== Forecasting Functions =============================================
# ========================================================================================================================================================================
# ========================================================================================================================================================================

import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import pickle
from scipy.signal import medfilt

# One auxiliary function
def import_RES_from_files(folder):
    import os
    # assign directory
    directory = folder+'RESDictionary_data/'
    keys2=[]
    RES2 = {}
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        print(filename[:-2])
        keys2.append(filename[:-2])
        RES2[filename[:-2]]=pickle.load(open(f,'rb'),encoding='latin1')
    return RES2

# ========================================================================================================================================================================
# ===================================================================== Normal forecasting function =================================================================
# ========================================================================================================================================================================

class SeismicityForecasting:
    def __init__(self,ForecastModel,Catalogue,CoulombStress,Mc=1.5,num_samples=50000,warmup_percentage=20,num_chains=1):
        
        # --- Defining the MCMC Parameters ---
        self.num_samples  = int(num_samples*((100-warmup_percentage)/100))
#        self.num_samples  = int(num_samples*((100-warmup_percentage)/100))
        self.warmup_steps = int(num_samples*(warmup_percentage/100))
        self.num_chains   = int(num_chains)

        # --- Training and Validation Info ---
        self.Training   = {}
        self.Validation = {}
        self.FullModel  = {}

        self.ForecastingModel   = ForecastModel

        # --- Initialising Observed Seismicity
        self._init_seismicity(Catalogue,Mc,CoulombStress)

        # --- Initialising Coulomb Stress Model ---
        self._init_ΔC(CoulombStress)

    ## def _init_seismicity(self,catalog,Mc): ## original version
    def _init_seismicity(self,catalog,Mc,CoulombStress): ## verison modified by Samson

        
        #Read the seismicity catalog  ## original version
        #tmpCat = pd.read_csv(catalog)
        #tmpCat.index = pd.to_datetime(tmpCat.time)
        #tmpCat = tmpCat[tmpCat['magnitude'].values >= Mc] # the filter
        #tmpCat['R']  = 1 # number of events per line of the catalog. This is used for the time averaging done later 
        #Average the seisms by time bins here yearly. If resampling is needed, the following lines need to be changed.
        #t_resampling = 'Y'
        #cat = tmpCat['R'].resample(t_resampling).sum()
        #cat.index = cat.index.year
        #self.Catalogue = {}
        #self.Catalogue['Dates'] = np.array(cat.index) #contains the dates of the resampled catalog
        #self.Catalogue['R']     = np.array(cat) #R contains the number of events per time bin

        #Read the seismicity catalog  ## modified version by Samson
        
        tmpCat = pd.read_csv(catalog)
        tmpCat = tmpCat[::-1] ## added SAmson
        tmpCat.time = pd.to_datetime(tmpCat.time,utc = True) ## added SAmson
        tmpCat = tmpCat[tmpCat['magnitude'].values >= Mc]
        tmpCat.reset_index(inplace = True, drop =True)
        t_ = []
        for i in range(len(tmpCat)):

            t_.append(tmpCat.time[i].year + tmpCat.time[i].day_of_year/365.25)
        t_ = np.asarray(t_)

        R_ = []
        vectime_ = CoulombStress['Time']
        dt_ = vectime_[-1] - vectime_[-2]
        fac_ = 1/dt_
        
        for i in range(len(vectime_)):

            if i < len(vectime_) - 1:

                t1 = vectime_[i]
                t2 = vectime_[i+1]
        
                esel = t_[(t_ >= t1 ) & (t_ < t2)]
        
                if len(esel) >= 1 : R_.append(len(esel))
                else : R_.append(0)

            else:

                t1 = vectime_[i]
                t2 = t1 + dt_
        
                esel = t_[(t_ >= t1 ) & (t_ < t2)]
        
                if len(esel) >= 1 : R_.append(len(esel))
                else : R_.append(0)


        #R_ = np.asarray(R_)
        R_ = np.asarray(R_)
#        R_ = medfilt(R_,3)
#        R_ = R_ * fac_ ## modified by samson: rate of eq per year
        self.Catalogue = {}
        self.Catalogue['Dates'] = vectime_ #contains the dates of the resampled catalog
        self.Catalogue['R']  = R_

        
    def _init_ΔC(self,CoulombStress):
        self.Coulomb = CoulombStress

    def run(self,training=[2020,2022],validation=[2022,2024],LLK='Gaussian'):

        RANDOM_SEED = 915623497 #random seed to be used for consistency between consecutive runs of the forecast
        np.random.seed(RANDOM_SEED)

        self.training_period    = training
        self.validation_period  = validation

        #  ------ Training using Metropolis MCMC -------
        #look for the index of the Coulomb dataset closest to the dates of the training period
        train_C = [abs(self.Coulomb['Time']-self.training_period[0]).argmin(),
                   abs(self.Coulomb['Time']-self.training_period[1]).argmin()]
        #look for the index of the observed seismicity dataset closest to the dates of the training period
        train_R = [abs(self.Catalogue['Dates']-self.training_period[0]).argmin(),
                   abs(self.Catalogue['Dates']-self.training_period[1]).argmin()]
        
        #gives coulomb_stress and seismicity rate to the ForecastingModel
        self.ForecastingModel.Coulomb    = self.Coulomb['Coulomb'][:,:,train_C[0]:train_C[1]]
        self.ForecastingModel.Robs       = self.Catalogue['R'][train_R[0]:train_R[1]]
        self.ForecastingModel.Robs_Dates = self.Catalogue['Dates'][train_R[0]:train_R[1]]
        if LLK == 'Poisson':
            self.ForecastingModel.create_model_Poisson()
        elif LLK == 'Gaussian':
            self.ForecastingModel.create_model()
        
        print('========= Training - Running MCMC =========')

        with self.ForecastingModel.model: #define the data for the MCMC
            pm.set_data({'dC':self.ForecastingModel.Coulomb,'Robs':self.ForecastingModel.Robs})
            step  = pm.Metropolis() #here the step is based on a Metropolis sampler algorithm to accept/reject the samples
            trace = pm.sample(draws=self.num_samples, chains=self.num_chains, tune=self.warmup_steps, step=step, random_seed=RANDOM_SEED,return_inferencedata=False)
#            trace = pm.sample(draws=self.num_samples, chains=self.num_chains, tune=self.warmup_steps, step=step, random_seed=RANDOM_SEED)
        
        print('Saving trace data')
        self.trace = trace

        print('========= Training - Determine R_Pred,Betas,logp =========')
        self.Training['Time'] = self.ForecastingModel.Robs_Dates
        self.Training['Rpred'],self.Training['betas'],self.Training['logp'] = self.ForecastingModel.forecast(self.trace,self.num_chains,self.num_samples,numsamples=1000,compute_logp=True) #Initially numsamples=1000


        # # # #  ------ Determining for Validation logp  -------
        print('========= Validation - Determine R_Pred,Betas,logp =========')
        val_C = [abs(self.Coulomb['Time']-self.validation_period[0]).argmin(),
                   abs(self.Coulomb['Time']-self.validation_period[1]).argmin()]
        val_R = [abs(self.Catalogue['Dates']-self.validation_period[0]).argmin(),
                   abs(self.Catalogue['Dates']-self.validation_period[1]).argmin()]

        self.ForecastingModel.Coulomb    = self.Coulomb['Coulomb'][:,:,val_C[0]:val_C[1]]
        self.ForecastingModel.Robs       = self.Catalogue['R'][val_R[0]:val_R[1]][:-1]
        self.ForecastingModel.Robs_Dates = self.Catalogue['Dates'][val_R[0]:val_R[1]][:-1]
        self.Validation['Time'] = self.ForecastingModel.Robs_Dates 

        #  --------- Determining the Earthquake rate for full period
        print('========= Full-time Evolution - Determine R_Pred =========')
        self.ForecastingModel.Coulomb    = self.Coulomb['Coulomb']
        #Rpred contains one sample of the seismicity rates for all the time period predicted for given parameters at one random iteration of the MCMC
        self.FullModel['Rpred'],self.FullModel['betas'] = self.ForecastingModel.forecast(self.trace,self.num_chains,self.num_samples,numsamples=1000,compute_logp=False)
        self.FullModel['Time']    = self.Coulomb['Time']
        self.FullModel['Robs']    = self.Catalogue['R'] #Observed seismicity rate
        self.FullModel['Robs_Ti'] = self.Catalogue['Dates'] #Time of each event
