#################### Functions for elastic stresses model ###################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import datetime
import pandas as pd
import math
from scipy.stats import kde
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
import okada_wrapper
import cutde.halfspace as HS
import cutde.fullspace as FS
import matplotlib.colors as mcolors
import pickle
from scipy.spatial import Delaunay
import networkx as nx


class PoroElasModel_HF:
    
    def cluster_wells(self,well_earthquake_map):
        """
        Groups fracking wells into families based on shared earthquakes and returns associated earthquakes.
        
        :param well_earthquake_map: Dictionary where keys are well IDs and values are sets of associated earthquake IDs.
        :return: List of tuples, where each tuple contains a set of connected wells and a set of associated earthquakes.
        """
        # Create a graph
        G = nx.Graph()
        
        # Reverse mapping: Create earthquake to well associations
        earthquake_well_map = {}
        for well, earthquakes in well_earthquake_map.items():
            for eq in earthquakes:
                if eq not in earthquake_well_map:
                    earthquake_well_map[eq] = set()
                earthquake_well_map[eq].add(well)
        
        # Add edges between wells that share at least one earthquake
        for wells in earthquake_well_map.values():
            wells = list(wells)  # Convert to list for indexing
            for i in range(len(wells)):
                for j in range(i + 1, len(wells)):
                    G.add_edge(wells[i], wells[j])
        
        # Extract connected components (families of wells)
        well_families = [set(component) for component in nx.connected_components(G)]
        
        # Associate earthquakes to each family
        family_earthquakes = []
        for family in well_families:
            associated_earthquakes = set()
            for well in family:
                associated_earthquakes.update(well_earthquake_map[well])
            family_earthquakes.append((family, associated_earthquakes))
        
        return family_earthquakes
    
    
    
    ############
    
    
    def distance(self,s_lat, s_lng, e_lat, e_lng):
        
        # approximate radius of earth in km
        R = 6371.0 * 1e3
        
        s_lat = s_lat*np.pi/180.0                      
        s_lng = np.deg2rad(s_lng)     
        e_lat = np.deg2rad(e_lat)                       
        e_lng = np.deg2rad(e_lng)  
        
        d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2
        
        return 2 * R * np.arcsin(np.sqrt(d)) 
    
    
    
    ############
    
    
    def slip_traction_component(self,tau_vector, strike_deg, dip_deg, rake_deg):
        """
        Projects the traction vector tau along the slip direction defined by strike, dip, and rake.
        
        Parameters:
        - tau_vector: array-like, shape (3,), traction vector [tau_N, tau_E, tau_D]
        - strike_deg: float, strike angle in degrees
        - dip_deg: float, dip angle in degrees
        - rake_deg: float, rake angle in degrees
        
        Returns:
        - tau_slip: float, component of traction along slip direction
        """
    
        # Convert angles to radians
        strike = np.radians(strike_deg)
        dip = np.radians(dip_deg)
        rake = np.radians(rake_deg)
        
        # Define slip direction vector in NED (North-East-Down)
        s_N = -np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike)
        s_E =  np.cos(rake) * np.cos(strike) - np.sin(rake) * np.cos(dip) * np.sin(strike)
        s_D =  np.sin(rake) * np.sin(dip)
        
        s_hat = np.array([s_N, s_E, s_D])
        
        # Normalize slip direction vector (optional, but should be unit vector)
        s_hat /= np.linalg.norm(s_hat)
        
        # Compute slip component of traction vector
        tau_slip = np.dot(tau_vector, s_hat)
        
        return tau_slip
    
    ##############
    
    def project_stress_to_fault(self,sigma, strike, dip):
    
        n = np.array([
           - np.sin(dip) * np.sin(strike),
            np.sin(dip) * np.cos(strike),
           -np.cos(dip)
        ])
        
        traction = np.dot(sigma,n)
        stress_fn = np.dot(np.dot(sigma,n),n)
        stress_sh   = np.sqrt(np.sum(np.dot(sigma,n)**2,axis=-1) - stress_fn**2)
        return stress_sh,stress_fn,traction
    
    
    #################
    
    def compute_maxCS_proj(self,stress_mat,strike,dip,rake = None):
    
        max_CS_proj = np.zeros([stress_mat.shape[0],stress_mat.shape[2]])
        mu = 0.6
    
    
        for i in range(stress_mat.shape[0]):
            for j in range(stress_mat.shape[2]):
    
                vec = stress_mat[i,:,j]
        
                Stress = np.zeros([3,3])
                Stress[0,0] = vec[0]                     # Sxx
                Stress[1,1] = vec[1]                     # Syy
                Stress[2,2] = vec[2]                      # Szz
                Stress[1,0] = Stress[0,1] = vec[3]   # Sxy
                Stress[2,0] = Stress[0,2] = vec[4]   # Sxz
                Stress[2,1] = Stress[1,2] = vec[5]  # Syz
        
                output_ = self.project_stress_to_fault(Stress, strike, dip)
    
                if rake is not None:
                    tau_s = self.slip_traction_component(output_[2], strike, dip, rake)
                    CS_FAULT = tau_s - mu*output_[1]
                else:
                    CS_FAULT = output_[0] - mu*output_[1]
    
                max_CS_proj[i,j] = CS_FAULT
                
        return max_CS_proj    
    
    ##########
    
    
    def compute_maxCS(self,stress_mat):
    
        max_CS = np.zeros([stress_mat.shape[0],stress_mat.shape[2]])
        mu = 0.6
    
        for i in range(stress_mat.shape[0]):
            print(i)
            
            for j in range(stress_mat.shape[2]):
    
                vec = stress_mat[i,:,j]

                if np.abs(stress_mat[i,:,j]).sum() > 0:
        
                    Stress = np.zeros([3,3])
                    Stress[0,0] = vec[0]                     # Sxx
                    Stress[1,1] = vec[1]                     # Syy
                    Stress[2,2] = vec[2]                      # Szz
                    Stress[1,0] = Stress[0,1] = vec[3]   # Sxy
                    Stress[2,0] = Stress[0,2] = vec[4]   # Sxz
                    Stress[2,1] = Stress[1,2] = vec[5]  # Syz
            
                    Stress_maxshear  = 0.5*(np.max(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1) - np.min(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1))
                    Stress_maxnormal = 0.5*(np.max(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1) + np.min(np.linalg.eigvals(np.nan_to_num(Stress)),axis=-1))
                    Stress_maxcoulomb = (Stress_maxshear - mu*(Stress_maxnormal))
        
                    max_CS[i,j] = Stress_maxcoulomb
                else : 
                    max_CS[i,j] = 0
                    
     
        return max_CS
    
    
    ##############
    
    
    ## kronecker detla function
    
    def k_r(self,i,j):
        if i == j : 
            output = 1
        else:
            output = 0
        return output 
    
    ##############
    
    
    def KGD(self,young_modulus,poisson_ratio,rock_toughness,vol_inj):
    
        ## return crack aperture based on KGD equations:: fracture toughness dominated regime
        
        young_modulus_prime = young_modulus / (1-poisson_ratio**2)
        
        R = (3*young_modulus_prime*vol_inj / (8 * np.sqrt(np.pi) * rock_toughness))**0.4
        p = np.sqrt(np.pi) * rock_toughness / (2 * np.sqrt(R))
        w = 8 * p * R / (np.pi * young_modulus_prime)
    
        return (R,p,w)
        
    
    ###########
    
    
    def stress_okada(self,obs_p,center_,width_,height_,poisson_ratio,shear_modulus,opening,direction):
        
        
        ## obs_p is the observers -- dimension = N x 3 (N observers and X,Y,Z)
        ## tris is the coordinates of the vertices of the two triangles which composed the plane (dimension 2,3,3)
        if direction == 'x':
    
            c1 = [0 , center_[1] - width_/2  , center_[2] - height_/2]
            c2 = [0 , center_[1] - width_/2 , center_[2] + height_/2]
            c3 = [0, center_[1] + width_/2  , center_[2] + height_/2]
            c4 = [0, center_[1] + width_/2  , center_[2] - height_/2]
            
        if direction == 'y':
            
            c1 = [center_[0] - width_/2 , 0 , center_[2] - height_/2]
            c2 = [center_[0] - width_/2 , 0 , center_[2] + height_/2]
            c3 = [center_[0] + width_/2 , 0 , center_[2] + height_/2]
            c4 = [center_[0] + width_/2 , 0 , center_[2] - height_/2]
    
        if direction == 'z':

        
            c1 = [center_[0] - height_/2 , center_[1] - width_/2  , center_[2]]
            c2 = [center_[0] + height_/2 , center_[1] - width_/2 , center_[2]]
            c3 = [center_[0] + height_/2, center_[1] + width_/2  , center_[2]]
            c4 = [center_[0] - height_/2, center_[1] + width_/2  , center_[2]]
    
    
        fault_pts = np.array([c1,c2,c3,c4])
        fault_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        tris = fault_pts[fault_tris]
    
        strain_mat = HS.strain_matrix(obs_pts=obs_p, tris=tris, nu=poisson_ratio)
        strain_mat = strain_mat.sum(axis = 2)[:,:,-1]
        strain_mat = strain_mat * opening #/(2 * np.pi)
    
        stress_mat = HS.strain_to_stress(strain_mat,shear_modulus,poisson_ratio)
    
        return -stress_mat
        
    
    ##########
    
    
    def rot_mat(self,x_,y_,theta_):
        try:
            l_ = len(x_)
            new_x,new_y = np.zeros(l_),np.zeros(l_)
            for i in range(l_):
                new_x[i] = x_[i] * np.cos(np.deg2rad(theta_)) - y_[i] * np.sin(np.deg2rad(theta_))
                new_y[i] = x_[i] * np.sin(np.deg2rad(theta_)) + y_[i] * np.cos(np.deg2rad(theta_))
        except:
            new_x = x_ * np.cos(np.deg2rad(theta_)) - y_ * np.sin(np.deg2rad(theta_))
            new_y = x_ * np.sin(np.deg2rad(theta_)) + y_ * np.cos(np.deg2rad(theta_))
            
        return new_x,new_y
    
    
    ######### 
    
    
    def datetime_to_years(self,datetime_):
        try:
            
            time_year = datetime_.dt.year + datetime_.dt.day_of_year/365.25 + datetime_.dt.hour/(365.25*24)

        except:

            time_year = datetime_.year + datetime_.day_of_year/365.25 + datetime_.hour/(365.25*24)
          
        return time_year
    
    ######### 
    
    
    def datetime_to_years_hf(self,datetime_):
        try:
            
            time_year = datetime_.dt.year + datetime_.dt.day_of_year/365.25 

        except:

            time_year = datetime_.year + datetime_.day_of_year/365.25 
          
        return time_year
        
    #########
    
    def select_eq_datafame(self,df_eq_select,df_ope,eq_time_years,dist_thresh,end_production=None):
    
        dataframe_ope = df_ope.copy()
        dataframe_ope.reset_index(drop = True)
        dataframe_eq = df_eq_select.copy()
        
        if dataframe_ope['BotLat'] > 0:
            
            lat_mean = (dataframe_ope['SurfLat'] + dataframe_ope['BotLat']) / 2
            lon_mean = (dataframe_ope['SurfLon'] + dataframe_ope['BotLon']) / 2
        else:
            
            lat_mean = dataframe_ope['SurfLat']
            lon_mean = dataframe_ope['SurfLon']
            
    
        start_inj = self.datetime_to_years(pd.to_datetime(dataframe_ope['JobStartDate']))
        if end_production: end_inj = end_production
        else: end_inj = self.datetime_to_years(pd.to_datetime(dataframe_ope['first_prod_date']))
    
        Lat_to_m = 111.139*1e3
        Lon_to_m = 111.139*1e3 * (np.cos(np.pi * lat_mean / 180))
    
        ##### center earthquakes ######
    
        lat_eq = dataframe_eq['latitude']
        lon_eq = dataframe_eq['longitude']
    
        lat_eq_centered = lat_eq - lat_mean
        lon_eq_centered = lon_eq - lon_mean
    
        x_eq_centered = lon_eq_centered * Lon_to_m
        y_eq_centered = lat_eq_centered * Lat_to_m
    
        
        if dataframe_ope['BotLat'] > 0:
            
            surflat_centered = dataframe_ope['SurfLat'] - lat_mean
            botlat_centered = dataframe_ope['BotLat'] - lat_mean
    
            surflon_centered = dataframe_ope['SurfLon'] - lon_mean
            botlon_centered = dataframe_ope['BotLon'] - lon_mean
    
            x_surf_centered = surflon_centered * Lon_to_m
            y_surf_centered = surflat_centered * Lat_to_m
    
            x_bot_centered = botlon_centered * Lon_to_m
            y_bot_centered = botlat_centered * Lat_to_m
            
            hypo_ = np.sqrt(x_surf_centered**2 + y_surf_centered**2)

            if hypo_ > 0:
            
                rotation_angle = - np.sign(x_surf_centered) * np.sign(y_surf_centered) * np.degrees(np.arccos(np.abs(x_surf_centered) / hypo_))
        
                x_top_rot,y_top_rot = self.rot_mat(x_surf_centered,y_surf_centered,rotation_angle)
                x_bot_rot,y_bot_rot = self.rot_mat(x_bot_centered,y_bot_centered,rotation_angle)
        
                x_eq_rot_,y_eq_rot_ = self.rot_mat(x_eq_centered,y_eq_centered,rotation_angle)
        
                max_x,min_x = np.max([x_top_rot,x_bot_rot]),np.min([x_top_rot,x_bot_rot])
        
                cond_ = (eq_time_years >= start_inj) & (eq_time_years <= end_inj) & (x_eq_rot_ <= max_x + dist_thresh) & (x_eq_rot_ >= min_x-dist_thresh) & (y_eq_rot_ >= -dist_thresh) & (y_eq_rot_ <= dist_thresh)
        
                df_sel = dataframe_eq.loc[cond_]

            else:
                
                    
                cond_ = (eq_time_years >= start_inj) & (eq_time_years <= end_inj) & (x_eq_centered <= dist_thresh) & (x_eq_centered >= dist_thresh) & (y_eq_centered >= -dist_thresh) & (y_eq_centered <= dist_thresh)
                df_sel = dataframe_eq.loc[cond_]
                     
    
        else:
    
            cond_ = (eq_time_years >= start_inj) & (eq_time_years <= end_inj) & (x_eq_centered <= dist_thresh) & (x_eq_centered >= dist_thresh) & (y_eq_centered >= -dist_thresh) & (y_eq_centered <= dist_thresh)
            df_sel = dataframe_eq.loc[cond_]
            
    
        return df_sel
        
    ##############

    def select_eq_datafame_refined(self,df_eq_select,df_ope,eq_time_years,dist_thresh,tstart,tend):
        
    
        dataframe_ope = df_ope.copy()
        dataframe_ope.reset_index(drop = True)
        dataframe_eq = df_eq_select.copy()
        
        if dataframe_ope['BotLat'] > 0:
            
            lat_mean = (dataframe_ope['SurfLat'] + dataframe_ope['BotLat']) / 2
            lon_mean = (dataframe_ope['SurfLon'] + dataframe_ope['BotLon']) / 2
        else:
            
            lat_mean = dataframe_ope['SurfLat']
            lon_mean = dataframe_ope['SurfLon']
            
    
        start_inj = tstart
        end_inj = tend
    
        Lat_to_m = 111.139*1e3
        Lon_to_m = 111.139*1e3 * (np.cos(np.pi * lat_mean / 180))
    
        ##### center earthquakes ######
    
        lat_eq = dataframe_eq['latitude']
        lon_eq = dataframe_eq['longitude']
    
        lat_eq_centered = lat_eq - lat_mean
        lon_eq_centered = lon_eq - lon_mean
    
        x_eq_centered = lon_eq_centered * Lon_to_m
        y_eq_centered = lat_eq_centered * Lat_to_m
    
        
        if dataframe_ope['BotLat'] > 0:
            
            surflat_centered = dataframe_ope['SurfLat'] - lat_mean
            botlat_centered = dataframe_ope['BotLat'] - lat_mean
    
            surflon_centered = dataframe_ope['SurfLon'] - lon_mean
            botlon_centered = dataframe_ope['BotLon'] - lon_mean
    
            x_surf_centered = surflon_centered * Lon_to_m
            y_surf_centered = surflat_centered * Lat_to_m
    
            x_bot_centered = botlon_centered * Lon_to_m
            y_bot_centered = botlat_centered * Lat_to_m
            
            hypo_ = np.sqrt(x_surf_centered**2 + y_surf_centered**2)

            if hypo_ > 0:
            
                rotation_angle = - np.sign(x_surf_centered) * np.sign(y_surf_centered) * np.degrees(np.arccos(np.abs(x_surf_centered) / hypo_))
        
                x_top_rot,y_top_rot = self.rot_mat(x_surf_centered,y_surf_centered,rotation_angle)
                x_bot_rot,y_bot_rot = self.rot_mat(x_bot_centered,y_bot_centered,rotation_angle)
        
                x_eq_rot_,y_eq_rot_ = self.rot_mat(x_eq_centered,y_eq_centered,rotation_angle)
        
                max_x,min_x = np.max([x_top_rot,x_bot_rot]),np.min([x_top_rot,x_bot_rot])
        
                cond_ = (eq_time_years >= start_inj) & (eq_time_years <= end_inj) & (x_eq_rot_ <= max_x + dist_thresh) & (x_eq_rot_ >= min_x-dist_thresh) & (y_eq_rot_ >= -dist_thresh) & (y_eq_rot_ <= dist_thresh)
        
                df_sel = dataframe_eq.loc[cond_]

            else:
                
                    
                cond_ = (eq_time_years >= start_inj) & (eq_time_years <= end_inj) & (x_eq_centered <= dist_thresh) & (x_eq_centered >= dist_thresh) & (y_eq_centered >= -dist_thresh) & (y_eq_centered <= dist_thresh)
                df_sel = dataframe_eq.loc[cond_]
                     
    
        else:
    
            cond_ = (eq_time_years >= start_inj) & (eq_time_years <= end_inj) & (x_eq_centered <= dist_thresh) & (x_eq_centered >= dist_thresh) & (y_eq_centered >= -dist_thresh) & (y_eq_centered <= dist_thresh)
            df_sel = dataframe_eq.loc[cond_]
            
    
        return df_sel
    

        
    ##############
    
    
    
    def stress_rot(self,mat_,theta):
        rot_matrix_3D = np.array([[np.cos(theta), -np.sin(theta),0],
                                [np.sin(theta),np.cos(theta),0],
                            [0,0,1]])
        
        output_ = np.zeros(mat_.shape)
        
        mat_Tmp = np.zeros([len(mat_),3,3])
        mat_Tmp[:,0,0] = mat_[:,0]
        mat_Tmp[:,1,1] = mat_[:,1]
        mat_Tmp[:,2,2] = mat_[:,2]
        mat_Tmp[:,0,1] = mat_Tmp[:,1,0] = mat_[:,3]
        mat_Tmp[:,0,2] = mat_Tmp[:,2,0] = mat_[:,4]
        mat_Tmp[:,1,2] = mat_Tmp[:,2,1] = mat_[:,5]
    
        mat_rot = rot_matrix_3D @ mat_Tmp @ rot_matrix_3D.T
    
        output_[:,0] = mat_rot[:,0,0]
        output_[:,1] = mat_rot[:,1,1]
        output_[:,2] = mat_rot[:,2,2]
        output_[:,3] = mat_rot[:,0,1]
        output_[:,4] = mat_rot[:,0,2]
        output_[:,5] = mat_rot[:,1,2]
    
        return output_
    
    
    #############
    
    
    def get_fault_nodes_strikedip(self,strike,dip, height, length, nodes):
        """
        Generate evenly spaced nodes on a fault plane.
    
        Parameters:
        strike (float): The strike of the fault in degrees.
        dip (float): The dip of the fault in degrees.
        vertical_extension (float): The vertical extent of the fault.
        width (float): The horizontal extent of the fault.
        node_spacing (float): The spacing between nodes in meters.
    
        Returns:
        numpy.ndarray: Coordinates of the nodes in the format (x, y, z).
        
        """
        
        # Calculate the number of nodes along the strike and dip directions
        x = np.linspace(-height/2,height/2,nodes)
        y = np.linspace(-length/2,length/2,nodes)
    
        xmesh,ymesh = np.meshgrid(x,y)
    
        xmesh_flatten,ymesh_flatten  =  xmesh.flatten(),ymesh.flatten()
    
        coor_fault = np.zeros([len(xmesh_flatten),3])
        coor_fault[:,0], coor_fault[:,1] = xmesh_flatten,ymesh_flatten
    
        dip = np.deg2rad(dip)
        strike = np.deg2rad(strike)
    
    
        Rx = np.array([[1,0,0],
                     [0,np.cos(dip),-np.sin(dip)],
                     [0,np.sin(dip),-np.cos(dip)]])
    
        Rx = np.array([[np.cos(dip),0,np.sin(dip)],
                     [0,1,0],
                     [-np.sin(dip),0,np.cos(dip)]])
    
        Rz = np.array([[np.cos(-strike),-np.sin(-strike),0],
                   [np.sin(-strike),np.cos(-strike),0],
                    [0,0,1]])
    
        coor_fault_rotate = coor_fault @ Rx.T @ Rz.T
        
        return coor_fault_rotate
        
        
    
    
    
    ###############
    
    
    
    ## main function to calculate poroelastic stresses due to a displacement discontinuity - equation E.144 page 836 (Cheng)
    ## note I did not include the heaviside function since I set thau = 0 
    def poroelastic_stresses_undrained(self,mu,poi_u,x,y,z,k,l):
    
        r = np.sqrt(x**2 + y**2 + z**2)
        r_x = x / r
        r_y = y / r
        r_z = z / r
        
        def r_d(ind_):
            
            if ind_ == 0 : output = r_x
            elif ind_ == 1 : output = r_y
            elif ind_ == 2 : output = r_z
    
            return output

        def k_r(i,j):
            if i == j : 
                output = 1
            else:
                output = 0
            return output 
    
    
        fac1 = mu/(4*np.pi*(1 - poi_u)*(r**3))
        
        ind_mat = (np.array([0,0]),
              np.array([1,1]),
              np.array([2,2]),
              np.array([0,1]),
              np.array([0,2]),
              np.array([1,2]))
        
        stress_mat = np.zeros([len(r),6])
    
        nn = 0
    
        for i in ind_mat:
    
            i,j = i[0],i[1]
    
            prod1 = (15*r_d(i)*r_d(j)*r_d(k)*r_d(l) - 
                     
                     (1-2*poi_u)*( k_r(k,j)*k_r(i,l) +  k_r(k,i)*k_r(j,l) - k_r(i,j)*k_r(k,l) + 3*k_r(k,l)*r_d(i)*r_d(j) +  3*k_r(i,j)*r_d(k)*r_d(l)) -
                     
                     poi_u*( 2*k_r(i,j)*k_r(k,l) + 3*k_r(l,j)*r_d(i)*r_d(k) + 3*k_r(i,l)*r_d(j)*r_d(k) + 3*k_r(k,j)*r_d(i)*r_d(l) + 3*k_r(i,k)*r_d(j)*r_d(l)) 
                    )
    
            
            
            prod_ = fac1 * prod1 
            stress_mat[:,nn] = prod_
            nn = nn +1
        return np.array(stress_mat)

   ###############
    
    
    
    ## main function to calculate poroelastic stresses due to a displacement discontinuity - equation E.144 page 836 (Cheng)
    ## note I did not include the heaviside function since I set thau = 0 
    def poroelastic_stresses(self,mu,poi_u,S,etha,diffu,x,y,z,k,l,t,thau):
    
        r = np.sqrt(x**2 + y**2 + z**2)
        r_x = x / r
        r_y = y / r
        r_z = z / r
    
        eps = r / np.sqrt(4*diffu*(t - thau))
    
        def r_d(ind_):
            
            if ind_ == 0 : output = r_x
            elif ind_ == 1 : output = r_y
            elif ind_ == 2 : output = r_z
    
            return output

        def k_r(self,i,j):
            if i == j : 
                output = 1
            else:
                output = 0
            return output 
    
    
        fac1 = mu/(4*np.pi*(1 - poi_u)*(r**3))
        fac2 = (etha**2) / (4 * np.pi * S * (r**3))
    
        ind_mat = (np.array([0,0]),
              np.array([1,1]),
              np.array([2,2]),
              np.array([0,1]),
              np.array([0,2]),
              np.array([1,2]))
        
        stress_mat = np.zeros([len(r),6])
    
        nn = 0
    
        for i in ind_mat:
    
            i,j = i[0],i[1]
    
            prod1 = (15*r_d(i)*r_d(j)*r_d(k)*r_d(l) - 
                     
                     (1-2*poi_u)*( k_r(k,j)*k_r(i,l) +  k_r(k,i)*k_r(j,l) - k_r(i,j)*k_r(k,l) + 3*k_r(k,l)*r_d(i)*r_d(j) +  3*k_r(i,j)*r_d(k)*r_d(l)) -
                     
                     poi_u*( 2*k_r(i,j)*k_r(k,l) + 3*k_r(l,j)*r_d(i)*r_d(k) + 3*k_r(i,l)*r_d(j)*r_d(k) + 3*k_r(k,j)*r_d(i)*r_d(l) + 3*k_r(i,k)*r_d(j)*r_d(l)) 
                    )
    
            
    
            prod2 = ( (8/np.sqrt(np.pi)) * (-2* k_r(i,j)*k_r(k,l) + 4*k_r(k,l)*r_d(i)*r_d(j) + 4*k_r(i,j)*r_d(k)*r_d(l) + 
                                            k_r(l,j)*r_d(k)*r_d(i) + k_r(l,i)*r_d(k)*r_d(j) + k_r(k,j)*r_d(l)*r_d(i) +
                                            k_r(k,i)*r_d(l)*r_d(j) - 10*r_d(i)*r_d(j)*r_d(k)*r_d(l)
                
            )*eps*np.exp(-eps**2) +
    
                     (16/np.sqrt(np.pi))*(-k_r(i,j)*k_r(k,l) + k_r(k,j)*r_d(i)*r_d(j) + k_r(i,j)*r_d(k)*r_d(l) - r_d(i)*r_d(j)*r_d(k)*r_d(l)
                         
                     )*(eps**3)*np.exp(-eps**2) +
    
                     3*(k_r(l,j)*k_r(k,i)+k_r(k,j)*k_r(i,l)+k_r(i,j)*k_r(k,l)-
                        5*k_r(l,j)*r_d(i)*r_d(k) - 5*k_r(l,i)*r_d(j)*r_d(k) - 5*k_r(k,j)*r_d(i)*r_d(l) - 
                        5*k_r(k,i)*r_d(j)*r_d(l) - 5*k_r(i,j)*r_d(k)*r_d(l) - 5*k_r(k,l)*r_d(i)*r_d(j) +
                        35*r_d(i)*r_d(j)*r_d(k)*r_d(l)
                         
                     )*(erf(eps)/(eps**2) - 2*np.exp(-eps**2)/(np.sqrt(np.pi)*eps)) +
    
                     2*(k_r(k,j)*k_r(i,l)+k_r(k,i)*k_r(j,l)-3*k_r(i,j)*k_r(k,l)+
                        3*k_r(k,l)*r_d(i)*r_d(j) + 3*k_r(i,j)*r_d(k)*r_d(l) -
                        3*k_r(i,l)*r_d(k)*r_d(j) - 3*k_r(k,j)*r_d(i)*r_d(l) - 
                        3*k_r(k,i)*r_d(l)*r_d(j) - 3*k_r(l,j)*r_d(k)*r_d(i) +
                        15*r_d(i)*r_d(j)*r_d(k)*r_d(l)
                         
                     )*erfc(eps)
                
            )
            
            prod_ = fac1 * prod1 + fac2 * prod2
            stress_mat[:,nn] = prod_
            nn = nn +1
        return np.array(stress_mat)

    
    ###############
    

    def pore_pressure_DC(self,S,etha,diffu,x,y,z,k,l,t,thau):
        
        def k_r(i,j):
            
            if i == j : 
                output = 1
            else:
                output = 0
            return output 
    
        r = np.sqrt(x**2 + y**2 + z**2)
        r_x = x / r
        r_y = y / r
        r_z = z / r
    
        eps = r / np.sqrt(4*diffu*(t - thau))
    
        coor = [x,y,z]
        r_k,r_l = coor[k]/r,coor[l]/r
    
        output_ = (etha/(2*np.pi*S*(r**3))) * (
            (k_r(k,l) - 3*r_k*r_l)*(erf(eps) - (2/np.sqrt(np.pi))*eps*np.exp(-(eps**2)))
            - ((4/np.sqrt(np.pi)) * ((k_r(k,l) - r_k*r_l) * (eps**3) * np.exp(-(eps**2))))
        )
        return np.array(output_)
        
    ###############

    def pore_pressure_undrained_DC(self,S,etha,diffu,x,y,z,k,l):

        def k_r(i,j):
            if i == j : 
                output = 1
            else:
                output = 0
            return output 
        
        r = np.sqrt(x**2 + y**2 + z**2)
        r_x = x / r
        r_y = y / r
        r_z = z / r
    
        coor = [x,y,z]
        r_k,r_l = coor[k]/r,coor[l]/r
    
        output_ = (etha/(2*np.pi*S*(r**3))) * (
            (k_r(k,l) - 3*r_k*r_l)
        )
        return np.array(output_)    
        
                
    
    ###############
    
    
    def interp_prod(self,vectime,vecrate):
        
        d_ = self.datetime_to_years(pd.to_datetime(vectime)).values
        newvecx= np.arange(d_.min(),d_.max(),7/365.24)
        newvecy = np.interp(newvecx,d_,vecrate.cumsum())
        return (np.array(newvecx),np.array(newvecy))

    
    ###############

    def flag_family_wells(self,df_hf):
    ### attribute to each well a number 
        
    
        ### 0 :  the well follows the following rule: startJob <= endJob <= completion_date <= first_production_date -- here we assume that flowback occurs
        # between endJob and compltetion_date
        
        ### 1 : the well follows the following rule: startJob <= first_production_date <= endJob -- here we assume that flowback has
        # has been included in production data
    
        ### 2 : the well follows the following rule: startJob <= endJob <= first_production_date <= completion_date  -- here we assume that
        #flow back occurs between endJob and first_production_date

        ### 2 : the well follows the following rule: startJob <= endJob <= first_production_date and  endJob <= completion_date  -- here we assume         that
        #flow back occurs between endJob and first_production_date
        
        df_copy = df_hf.copy()
        L_ = len(df_copy)
    
        t1 = self.datetime_to_years_hf(pd.to_datetime(df_copy['JobStartDate'])).values
        t2 = self.datetime_to_years_hf(pd.to_datetime(df_copy['JobEndDate'])).values
        t3 = self.datetime_to_years_hf(pd.to_datetime(df_copy['completion_date'])).values
        t4 = self.datetime_to_years_hf(pd.to_datetime(df_copy['first_prod_date'])).values
    
        flag_ = np.nan * np.ones(len(df_hf))   
    
        for i in range(L_):
    
            t1_tmp = t1[i]
            t2_tmp = t2[i]
            t3_tmp = t3[i]
            t4_tmp = t4[i]
    
            cond_1 = t1_tmp <= t2_tmp < t3_tmp <= t4_tmp
            cond_2 = t1_tmp <= t4_tmp <= t2_tmp
            cond_3 = t1_tmp <= t2_tmp <= t4_tmp <= t3_tmp
            cond_4 = (t1_tmp <= t2_tmp <= t4_tmp) & (t3_tmp < t2_tmp)
            cond_5 = t1_tmp <= t2_tmp == t3_tmp <= t4_tmp
    
            if cond_1: flag_[i] =  0
            elif cond_2: flag_[i] =  1
            elif cond_3: flag_[i] =  2
            elif cond_4: flag_[i] =  2
            elif cond_5: flag_[i] =  1
    
        return flag_

                                    
    
    ###############

    def clean_catalog(self,dataframe_):
    
        df_hf = dataframe_.copy()
        df_hf.loc[:,'completion_date'] = pd.to_datetime(df_hf.loc[:,'completion_date'])
        df_hf.loc[:,'JobEndDate'] = pd.to_datetime(df_hf.loc[:,'JobEndDate'])
        
        df_hf.reset_index(drop = True,inplace = True)
    
        day_in_year = 1/365.25
        
    
        comple_year = pd.to_datetime(df_hf['completion_date']).dt.year.values + pd.to_datetime(df_hf['completion_date']).dt.day_of_year.values/365.25
        end_inj_year = pd.to_datetime(df_hf['JobEndDate']).dt.year.values + pd.to_datetime(df_hf['JobEndDate']).dt.day_of_year.values/365.25

        for i in range(len(df_hf)):
            
            if comple_year[i] == end_inj_year[i] : 
                df_hf.loc[i,'completion_date'] = df_hf.loc[i,'completion_date'] + pd.Timedelta(days = 0)
            else:
                toto = 0
        return df_hf 


    ##################

    def compute_CS_cheng_undrained(self,dataframe_injection,grid,mecha_param,number_fracture,width,height,time_vector,depth,prod_inj,
                                   ratio_flowback,ratio_proppant,flag):

        mu = mecha_param['shear_modulus']
        poi_u = mecha_param['poisson_ratio']
        S = mecha_param['S']
        etha = mecha_param['etha']
        diffu = mecha_param['diffu']

        #########  START ###########
    
        df_hf = dataframe_injection.copy()
        df_hf.reset_index(drop = True)
        
        
    
        #### define mean lat and min lon of the well
    
        lat_mean = (df_hf['SurfLat'] + df_hf['BotLat']) / 2
        lon_mean = (df_hf['SurfLon'] + df_hf['BotLon']) / 2
    
        ## reshape grid if necessary
    
        if len(grid.shape) != 2: grid.reshape((grid.shape[0]**2,grid.shape[-1]))
    
        ## define matrix for data stotage
    
        output_ = np.zeros([grid.shape[0],6,len(time_vector)])
    
        ##### extract depth of the well
    
        well_depth = - df_hf['Depth(m)']
    
        Lat_to_m = 111.139*1e3
        Lon_to_m = 111.139*1e3 * (np.cos(np.pi * lat_mean / 180))
    
        ## concert time_vector years
    
        time_vector_years = self.datetime_to_years_hf(pd.to_datetime(time_vector))
    
        ### define the occurence times in year of the eqs and the start and end of the ope
    
        start_inj = self.datetime_to_years_hf(pd.to_datetime(df_hf['JobStartDate']))
        end_inj = self.datetime_to_years_hf(pd.to_datetime(df_hf['JobEndDate']))
        completion_end = self.datetime_to_years_hf(pd.to_datetime(df_hf['completion_date']))
        start_prod = self.datetime_to_years_hf(pd.to_datetime(df_hf['first_prod_date']))
            
        duration_fracking = end_inj - start_inj
        duration_completion = completion_end - start_inj
        duration_prod = time_vector_years[-1] - start_prod
    
        ## define injected volume
        
        tot_inj_vol = df_hf['TotalVolumeInjected(l)']/1e3 ## in m^3
    
        ## define area of dislocation
    
        area_dislo = width * height 
        
        ##### center earthquakes ######
    
        lon_grid = grid[:,0]
        lat_grid = grid[:,1]
    
        lon_grid_centered = lon_grid - lon_mean
        lat_grid_centered = lat_grid - lat_mean
    
        x_grid_centered = lon_grid_centered * Lon_to_m
        y_grid_centered = lat_grid_centered * Lat_to_m
    
        cart_grid = np.zeros([len(x_grid_centered),3])
        cart_grid[:,0],cart_grid[:,1],cart_grid[:,2] = x_grid_centered,y_grid_centered,-np.ones(len(cart_grid))*depth
    
        ##### center well ######
    
        surflon_centered,botlon_centered = df_hf['SurfLon'] - lon_mean , df_hf['BotLon'] - lon_mean
        surflat_centered,botlat_centered = df_hf['SurfLat'] - lat_mean , df_hf['BotLat'] - lat_mean
    
        x_surf_centered,x_bot_centered = surflon_centered * Lon_to_m,botlon_centered * Lon_to_m
        y_surf_centered,y_bot_centered = surflat_centered * Lat_to_m,botlat_centered * Lat_to_m
    
        ##### rotate well so its horizontal axis aligns with x
    
        hypo_ = np.sqrt(x_surf_centered**2 + y_surf_centered**2)
        rotation_angle =  - np.sign(x_surf_centered) * np.sign(y_surf_centered) * np.arccos(np.abs(x_surf_centered) / hypo_)
        
        rot_matrix_2D = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle),np.cos(rotation_angle)]])
    
        mat_well = np.array([[x_surf_centered,y_surf_centered],
                          [x_bot_centered,y_bot_centered]])
        mat_well_rotate = mat_well @ rot_matrix_2D.T
    
        xsurf_rot,xbot_rot = mat_well_rotate[0,0],mat_well_rotate[1,0]
    
        ##### rotate grid #####
    
        cart_grid_rotate = cart_grid
        cart_grid_rotate[:,:2] = cart_grid_rotate[:,:2] @ rot_matrix_2D.T
    
        ###### find time_vector indexex for which well was active
    
        dt_ = np.diff(time_vector_years).min()
    
            #### mechanical parameters for okada's solutions
    
        poisson_ratio = mecha_param['poisson_ratio']
        shear_modulus = mecha_param['shear_modulus']
    
    
        ### define well loc
    
        x_loc_fracture = np.linspace(xbot_rot,xsurf_rot,number_fracture)
    
        ## 
    
        time_between_stages = (end_inj - start_inj) / number_fracture

        ## 
        k,l = 0,0
        
    
    ############# stress calculation ###########
    
    ### flag = 0 : startFrac <= endFrac <= completion_date <= production date    
        if flag == 0 :
            
            try:
                ind_frack = list(np.where((time_vector_years >= start_inj))[0])
            except:
                ind_frack = []
                
            ind_flowback = list(np.where((time_vector_years >= end_inj))[0])
            ind_prod = list(np.where((time_vector_years >= start_prod))[0])
    
            ind_flowbackend = np.where((time_vector_years - completion_end) >=0)[0][0]
            ind_frackend = np.where((time_vector_years - end_inj) >= 0)[0][0]

            ##### calculate induced stresses after fracking has been completed

            
            nfrac_select = len(x_loc_fracture)              
            tot_vol_per_frac = tot_inj_vol  / nfrac_select

            tot_stress_frac = np.zeros([grid.shape[0],6])
                
            for j in range(nfrac_select):

                center_ = np.array([x_loc_fracture[j],0,well_depth])
                x = cart_grid_rotate[:,0] - center_[0]
                y = cart_grid_rotate[:,1] - center_[1]
                z = cart_grid_rotate[:,2] - center_[2]
                
                sol2_ = tot_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                tot_stress_frac += sol2_rot_
  
                
            ##### for time between start_inj and end_inj
    
            for i in ind_frack:
                
                if i <= ind_frackend:
        
                    time_select = time_vector_years[i]
                    
                    ratio_ = (time_select - start_inj) / (end_inj - start_inj)
                    
                    nfrac_select = np.max([1,int(np.round(ratio_*len(x_loc_fracture)))])
                    nfrac_select = np.min([nfrac_select,len(x_loc_fracture)])
                         
                    frac_vol_per_frac = ratio_ * tot_inj_vol / nfrac_select
            
                    for j in range(nfrac_select):
                
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1]
                        z = cart_grid_rotate[:,2] - center_[2]
                        
                        sol2_ = frac_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                        sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                                    
                        output_[:,:,i] += sol2_rot_
                    
                        
                else:
                    
                    output_[:,:,i] = tot_stress_frac
                    
            
            # ########## flowbackkkkk
                
            for i in ind_flowback:
                             
                if i <= ind_flowbackend:
                        
            
                    time_select = time_vector_years[i]
                            
                    ratio_ = (time_select - end_inj) / (completion_end - end_inj)
                    ratio_ = np.min([1,ratio_])
                    nfrac_select = len(x_loc_fracture)
                                 
                    fb_vol_per_frac = ratio_flowback * ratio_ * tot_inj_vol  / nfrac_select
                    
                    for j in range(nfrac_select):
                                
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1]
                        z = cart_grid_rotate[:,2] - center_[2]
                        
                        sol2_ = fb_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                        sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                                    
                        output_[:,:,i] -= sol2_rot_
        
                                   
                else:

                    output_[:,:,i] -= ratio_flowback * tot_stress_frac
                    
        ########## production
                        
            if prod_inj[1].sum() > 0:
                
                for i in ind_prod:
                                
                    time_select = time_vector_years[i]
                    nfrac_select = len(x_loc_fracture)
                        
                    vectimeprod = prod_inj[0]
                    ind_ = np.argmin(np.abs(vectimeprod - time_select))
                    prodwater = prod_inj[1][ind_]/1e3 ## volume
                    if prodwater >= (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol : prodwater = (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol
                    prod_vol_per_frac = prodwater / nfrac_select                    
    
                    for j in range(nfrac_select):
                        
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1]
                        z = cart_grid_rotate[:,2] - center_[2]
                                        
                        sol2_ = prod_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                        sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                        output_[:,:,i] -= sol2_rot_
    
        
    
    ### flag = 1 : startFrac <= production <= endFrac
        
        elif flag == 1 :
            
            
            
            ind_frack = list(np.where((time_vector_years >= start_inj))[0])
            ind_prod = list(np.where((time_vector_years >= start_prod))[0])
            ind_frackend = np.where((time_vector_years - end_inj) >= 0)[0][0]

            ##### calculate induced stresses after fracking has been completed

            
            nfrac_select = len(x_loc_fracture)              
            tot_vol_per_frac = tot_inj_vol  / nfrac_select

            tot_stress_frac = np.zeros([grid.shape[0],6])
                
            for j in range(nfrac_select):

                center_ = np.array([x_loc_fracture[j],0,well_depth])
                x = cart_grid_rotate[:,0] - center_[0]
                y = cart_grid_rotate[:,1] - center_[1]
                z = cart_grid_rotate[:,2] - center_[2]
                
                sol2_ = tot_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                tot_stress_frac += sol2_rot_

            for i in ind_frack:
                
                
                if i <= ind_frackend:
        
                    time_select = time_vector_years[i]
                    
                    ratio_ = (time_select - start_inj) / (end_inj - start_inj)
                    
                    nfrac_select = np.max([1,int(np.round(ratio_*len(x_loc_fracture)))])
                    nfrac_select = np.min([nfrac_select,len(x_loc_fracture)])
                         
                    frac_vol_per_frac = ratio_ * tot_inj_vol / nfrac_select
            
                    for j in range(nfrac_select):
                        
                
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1]
                        z = cart_grid_rotate[:,2] - center_[2]
                        
                        sol2_ = frac_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                        sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                                    
                        output_[:,:,i] += sol2_rot_
                    
                        
                else:
                    
                    output_[:,:,i] = tot_stress_frac

                    
        
            ######## production
            
                        
            if prod_inj[1].sum() > 0:
                
                for i in ind_prod:
                                
                    time_select = time_vector_years[i]
                    nfrac_select = len(x_loc_fracture)
                        
                    vectimeprod = prod_inj[0]
                    ind_ = np.argmin(np.abs(vectimeprod - time_select))
                    prodwater = prod_inj[1][ind_]/1e3 ## volume
                    if prodwater >= (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol : prodwater = (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol
                    prod_vol_per_frac = prodwater / nfrac_select                    
    
                    for j in range(nfrac_select):
                        
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1]
                        z = cart_grid_rotate[:,2] - center_[2]
                                        
                        sol2_ = prod_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                        sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                        output_[:,:,i] -= sol2_rot_
    
    ### flag = 2 : startFrac <= endFrac <= production_date <= completion_date date    
        elif flag == 2 :
            
            ind_frack = list(np.where((time_vector_years >= start_inj))[0])
            ind_flowback = list(np.where((time_vector_years >= end_inj))[0])
            ind_prod = list(np.where((time_vector_years >= start_prod))[0])
    
            ind_flowbackend = np.where((time_vector_years - start_prod) >=0)[0][0]
            ind_frackend = np.where((time_vector_years - end_inj) >= 0)[0][0]
            
            ##### calculate induced stresses after fracking has been completed

            
            nfrac_select = len(x_loc_fracture)              
            tot_vol_per_frac = tot_inj_vol  / nfrac_select

            tot_stress_frac = np.zeros([grid.shape[0],6])
                
            for j in range(nfrac_select):

                center_ = np.array([x_loc_fracture[j],0,well_depth])
                x = cart_grid_rotate[:,0] - center_[0]
                y = cart_grid_rotate[:,1] - center_[1]
                z = cart_grid_rotate[:,2] - center_[2]
                
                sol2_ = tot_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                tot_stress_frac += sol2_rot_
  
                
            ##### for time between start_inj and end_inj
    
            for i in ind_frack:
                
                if i <= ind_frackend:
        
                    time_select = time_vector_years[i]
                    
                    ratio_ = (time_select - start_inj) / (end_inj - start_inj)
                    
                    nfrac_select = np.max([1,int(np.round(ratio_*len(x_loc_fracture)))])
                    nfrac_select = np.min([nfrac_select,len(x_loc_fracture)])
                         
                    frac_vol_per_frac = ratio_ * tot_inj_vol / nfrac_select
            
                    for j in range(nfrac_select):
                        
                
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1]
                        z = cart_grid_rotate[:,2] - center_[2]
                        
                        sol2_ = frac_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                        sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                                    
                        output_[:,:,i] += sol2_rot_
                    
                        
                else:
                    
                    output_[:,:,i] = tot_stress_frac
                    
            
            for i in ind_flowback:
                             
                if i <= ind_flowbackend:
                        
            
                    time_select = time_vector_years[i]
                            
                    ratio_ = (time_select - end_inj) / (start_prod - end_inj)
                    ratio_ = np.min([1,ratio_])
                    nfrac_select = len(x_loc_fracture)
                                 
                    fb_vol_per_frac = ratio_flowback * ratio_ * tot_inj_vol  / nfrac_select
                    
                    for j in range(nfrac_select):
                                
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1]
                        z = cart_grid_rotate[:,2] - center_[2]
                        
                        sol2_ = fb_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                        sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                                    
                        output_[:,:,i] -= sol2_rot_
        
                                   
                else:

                    output_[:,:,i] -= ratio_flowback * tot_stress_frac
                    
        ########## production
                        
            if prod_inj[1].sum() > 0:
                
                for i in ind_prod:
                                
                    time_select = time_vector_years[i]
                    nfrac_select = len(x_loc_fracture)
                        
                    vectimeprod = prod_inj[0]
                    ind_ = np.argmin(np.abs(vectimeprod - time_select))
                    prodwater = prod_inj[1][ind_]/1e3 ## volume
                    if prodwater >= (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol : prodwater = (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol
                    prod_vol_per_frac = prodwater / nfrac_select                    
    
                    for j in range(nfrac_select):
                        
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1]
                        z = cart_grid_rotate[:,2] - center_[2]
                                        
                        sol2_ = prod_vol_per_frac * self.poroelastic_stresses_undrained(mu,poi_u,x,y,z,k,l)
                        sol2_rot_ = self.stress_rot(sol2_,-rotation_angle)
                        output_[:,:,i] -= sol2_rot_
    

        return output_


    def compute_pore_pressure_undrained(self,dataframe_injection,grid,mecha_param,number_fracture,width,height,time_vector,depth,prod_inj,
                                   ratio_flowback,ratio_proppant,flag):

        S = mecha_param['S']
        etha = mecha_param['etha']
        diffu =  mecha_param['diffu']

        #########  START ###########
    
        df_hf = dataframe_injection.copy()
        df_hf.reset_index(drop = True)
        
        
    
        #### define mean lat and min lon of the well
    
        lat_mean = (df_hf['SurfLat'] + df_hf['BotLat']) / 2
        lon_mean = (df_hf['SurfLon'] + df_hf['BotLon']) / 2
    
        ## reshape grid if necessary
    
        if len(grid.shape) != 2: grid.reshape((grid.shape[0]**2,grid.shape[-1]))
    
        ## define matrix for data stotage
    
        output_ = np.zeros([grid.shape[0],len(time_vector)])
    
        ##### extract depth of the well
    
        well_depth = - df_hf['Depth(m)']
    
        Lat_to_m = 111.139*1e3
        Lon_to_m = 111.139*1e3 * (np.cos(np.pi * lat_mean / 180))
    
        ## concert time_vector years
    
        time_vector_years = self.datetime_to_years_hf(pd.to_datetime(time_vector))
    
        ### define the occurence times in year of the eqs and the start and end of the ope
    
        start_inj = self.datetime_to_years_hf(pd.to_datetime(df_hf['JobStartDate']))
        end_inj = self.datetime_to_years_hf(pd.to_datetime(df_hf['JobEndDate']))
        completion_end = self.datetime_to_years_hf(pd.to_datetime(df_hf['completion_date']))
        start_prod = self.datetime_to_years_hf(pd.to_datetime(df_hf['first_prod_date']))
            
        duration_fracking = end_inj - start_inj
        duration_completion = completion_end - start_inj
        duration_prod = time_vector_years[-1] - start_prod
    
        ## define injected volume
        
        tot_inj_vol = df_hf['TotalVolumeInjected(l)']/1e3 ## in m^3
    
        ## define area of dislocation
    
        area_dislo = width * height 
        
        ##### center earthquakes ######
    
        lon_grid = grid[:,0]
        lat_grid = grid[:,1]
    
        lon_grid_centered = lon_grid - lon_mean
        lat_grid_centered = lat_grid - lat_mean
    
        x_grid_centered = lon_grid_centered * Lon_to_m
        y_grid_centered = lat_grid_centered * Lat_to_m
    
        cart_grid = np.zeros([len(x_grid_centered),3])
        cart_grid[:,0],cart_grid[:,1],cart_grid[:,2] = x_grid_centered,y_grid_centered,-np.ones(len(cart_grid))*depth
    
        ##### center well ######
    
        surflon_centered,botlon_centered = df_hf['SurfLon'] - lon_mean , df_hf['BotLon'] - lon_mean
        surflat_centered,botlat_centered = df_hf['SurfLat'] - lat_mean , df_hf['BotLat'] - lat_mean
    
        x_surf_centered,x_bot_centered = surflon_centered * Lon_to_m,botlon_centered * Lon_to_m
        y_surf_centered,y_bot_centered = surflat_centered * Lat_to_m,botlat_centered * Lat_to_m
    
        ##### rotate well so its horizontal axis aligns with x
    
        hypo_ = np.sqrt(x_surf_centered**2 + y_surf_centered**2)
        rotation_angle =  - np.sign(x_surf_centered) * np.sign(y_surf_centered) * np.arccos(np.abs(x_surf_centered) / hypo_)
        
        rot_matrix_2D = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle),np.cos(rotation_angle)]])
    
        mat_well = np.array([[x_surf_centered,y_surf_centered],
                          [x_bot_centered,y_bot_centered]])
        mat_well_rotate = mat_well @ rot_matrix_2D.T
    
        xsurf_rot,xbot_rot = mat_well_rotate[0,0],mat_well_rotate[1,0]
    
        ##### rotate grid #####
    
        cart_grid_rotate = cart_grid
        cart_grid_rotate[:,:2] = cart_grid_rotate[:,:2] @ rot_matrix_2D.T
    
        ###### find time_vector indexex for which well was active
    
        dt_ = np.diff(time_vector_years).min()
    
            #### mechanical parameters for okada's solutions
    
        poisson_ratio = mecha_param['poisson_ratio']
        shear_modulus = mecha_param['shear_modulus']
    
    
        ### define well loc
    
        x_loc_fracture = np.linspace(xbot_rot,xsurf_rot,number_fracture)
    
        ## 
    
        time_between_stages = (end_inj - start_inj) / number_fracture

        ##

        k,l = 0,0
        
    
    ############# stress calculation ###########
    
    ### flag = 0 : startFrac <= endFrac <= completion_date <= production date    
        if flag == 0 :
            
            ind_frack = list(np.where((time_vector_years >= start_inj))[0])
            ind_flowback = list(np.where((time_vector_years >= end_inj))[0])
            ind_prod = list(np.where((time_vector_years >= start_prod))[0])
    
            ind_flowbackend = np.where((time_vector_years - completion_end) >=0)[0][0]
            ind_frackend = np.where((time_vector_years - end_inj) >= 0)[0][0]

            ##### calculate induced stresses after fracking has been completed

            
            nfrac_select = len(x_loc_fracture)              
            tot_vol_per_frac = tot_inj_vol/ nfrac_select

            tot_pp_frac = np.zeros([grid.shape[0]])
                
            for j in range(nfrac_select):

                center_ = np.array([x_loc_fracture[j],0,well_depth])
                x = cart_grid_rotate[:,0] - center_[0]
                y = cart_grid_rotate[:,1] - center_[1] 
                z = cart_grid_rotate[:,2] - center_[2]
                sol_ = tot_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                tot_pp_frac += sol_
  
                
            ##### for time between start_inj and end_inj
    
            for i in ind_frack:
                
                if i <= ind_frackend:
        
                    time_select = time_vector_years[i]
                    
                    ratio_ = (time_select - start_inj) / (end_inj - start_inj)
                    
                    nfrac_select = np.max([1,int(np.round(ratio_*len(x_loc_fracture)))])
                    nfrac_select = np.min([nfrac_select,len(x_loc_fracture)])
                         
                    frac_vol_per_frac = ratio_ * tot_inj_vol / nfrac_select
            
                    for j in range(nfrac_select):
            
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1] 
                        z = cart_grid_rotate[:,2] - center_[2]
                        sol_ = frac_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                        output_[:,i] += sol_
                            
                else:
                    
                    output_[:,i] = tot_pp_frac
                    
            
            # ########## flowbackkkkk
                
            for i in ind_flowback:
                             
                if i <= ind_flowbackend:
                        
            
                    time_select = time_vector_years[i]
                            
                    ratio_ = (time_select - end_inj) / (completion_end - end_inj)
                    ratio_ = np.min([1,ratio_])
                    nfrac_select = len(x_loc_fracture)
                                 
                    fb_vol_per_frac = ratio_flowback * ratio_ * tot_inj_vol / nfrac_select
                    
                    for j in range(nfrac_select):
                        
                                
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1] 
                        z = cart_grid_rotate[:,2] - center_[2]
                        sol_ = fb_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                        output_[:,i] -= sol_
        
                                   
                else:

                    output_[:,i] -= ratio_flowback * tot_pp_frac
                    
        ######### production
                        
            if prod_inj[1].sum() > 0:
                
                for i in ind_prod:
                                
                    time_select = time_vector_years[i]
                    nfrac_select = len(x_loc_fracture)
                        
                    vectimeprod = prod_inj[0]
                    ind_ = np.argmin(np.abs(vectimeprod - time_select))
                    prodwater = prod_inj[1][ind_]/1e3 ## volume
                    if prodwater >= (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol : prodwater = (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol
                    prod_vol_per_frac = prodwater / nfrac_select                    
    
                    for j in range(nfrac_select):
                        
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1] 
                        z = cart_grid_rotate[:,2] - center_[2]
                        sol_ = prod_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                        output_[:,i] -= sol_
        
    
        
    
    ### flag = 1 : startFrac <= production <= endFrac
        
        elif flag == 1 :
            
            
            
            ind_frack = list(np.where((time_vector_years >= start_inj))[0])
            ind_prod = list(np.where((time_vector_years >= start_prod))[0])
            ind_frackend = np.where((time_vector_years - end_inj) >= 0)[0][0]

            ##### calculate induced stresses after fracking has been completed

            
            nfrac_select = len(x_loc_fracture)              
            tot_vol_per_frac = tot_inj_vol/ nfrac_select

            tot_pp_frac = np.zeros([grid.shape[0]])
                
            for j in range(nfrac_select):

                center_ = np.array([x_loc_fracture[j],0,well_depth])
                x = cart_grid_rotate[:,0] - center_[0]
                y = cart_grid_rotate[:,1] - center_[1] 
                z = cart_grid_rotate[:,2] - center_[2]
                sol_ = tot_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                tot_pp_frac += sol_

            for i in ind_frack:
                
                if i <= ind_frackend:
        
                    time_select = time_vector_years[i]
                    
                    ratio_ = (time_select - start_inj) / (end_inj - start_inj)
                    
                    nfrac_select = np.max([1,int(np.round(ratio_*len(x_loc_fracture)))])
                    nfrac_select = np.min([nfrac_select,len(x_loc_fracture)])
                         
                    frac_vol_per_frac = ratio_ * tot_inj_vol / nfrac_select
            
                    for j in range(nfrac_select):
            
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1] 
                        z = cart_grid_rotate[:,2] - center_[2]
                        sol_ = frac_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                        output_[:,i] += sol_
                            
                else:
                    
                    output_[:,i] = tot_pp_frac
        
        ########## production
                        
            if prod_inj[1].sum() > 0:
                
                for i in ind_prod:
                                
                    time_select = time_vector_years[i]
                    nfrac_select = len(x_loc_fracture)
                        
                    vectimeprod = prod_inj[0]
                    ind_ = np.argmin(np.abs(vectimeprod - time_select))
                    prodwater = prod_inj[1][ind_]/1e3 ## volume
                    if prodwater >= (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol : prodwater = (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol
                    prod_vol_per_frac = prodwater / nfrac_select                    
    
                    for j in range(nfrac_select):
                        
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1] 
                        z = cart_grid_rotate[:,2] - center_[2]
                        sol_ = prod_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                        output_[:,i] -= sol_
                    
    ### flag = 2 : startFrac <= endFrac <= production_date <= completion_date date    
        elif flag == 2 :
            
            ind_frack = list(np.where((time_vector_years >= start_inj))[0])
            ind_flowback = list(np.where((time_vector_years >= end_inj))[0])
            ind_prod = list(np.where((time_vector_years >= start_prod))[0])
    
            ind_flowbackend = np.where((time_vector_years - start_prod) >=0)[0][0]
            ind_frackend = np.where((time_vector_years - end_inj) >= 0)[0][0]

            ##### calculate induced stresses after fracking has been completed

            
            nfrac_select = len(x_loc_fracture)              
            tot_vol_per_frac = tot_inj_vol/ nfrac_select

            tot_pp_frac = np.zeros([grid.shape[0]])
                
            for j in range(nfrac_select):

                center_ = np.array([x_loc_fracture[j],0,well_depth])
                x = cart_grid_rotate[:,0] - center_[0]
                y = cart_grid_rotate[:,1] - center_[1] 
                z = cart_grid_rotate[:,2] - center_[2]
                sol_ = tot_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                tot_pp_frac += sol_
  
                
            ##### for time between start_inj and end_inj
    
            for i in ind_frack:
                
                if i <= ind_frackend:
        
                    time_select = time_vector_years[i]
                    
                    ratio_ = (time_select - start_inj) / (end_inj - start_inj)
                    
                    nfrac_select = np.max([1,int(np.round(ratio_*len(x_loc_fracture)))])
                    nfrac_select = np.min([nfrac_select,len(x_loc_fracture)])
                         
                    frac_vol_per_frac = ratio_ * tot_inj_vol / nfrac_select
            
                    for j in range(nfrac_select):
            
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1] 
                        z = cart_grid_rotate[:,2] - center_[2]
                        sol_ = frac_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                        output_[:,i] += sol_
                            
                else:
                    
                    output_[:,i] = tot_pp_frac
                    
            
            # ########## flowbackkkkk
                
            for i in ind_flowback:
                             
                if i <= ind_flowbackend:
                        
            
                    time_select = time_vector_years[i]
                            
                    ratio_ = (time_select - end_inj) / (start_prod - end_inj)
                    ratio_ = np.min([1,ratio_])
                    nfrac_select = len(x_loc_fracture)
                                 
                    fb_vol_per_frac = ratio_flowback * ratio_ * tot_inj_vol /  nfrac_select
                    
                    for j in range(nfrac_select):
                                
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1] 
                        z = cart_grid_rotate[:,2] - center_[2]
                        sol_ = fb_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                        output_[:,i] -= sol_
                                   
                else:

                    output_[:,i] -= ratio_flowback * tot_pp_frac
                    
        ########## production
            
            if prod_inj[1].sum() > 0:
                
                for i in ind_prod:
                                
                    time_select = time_vector_years[i]
                    nfrac_select = len(x_loc_fracture)
                        
                    vectimeprod = prod_inj[0]
                    ind_ = np.argmin(np.abs(vectimeprod - time_select))
                    prodwater = prod_inj[1][ind_]/1e3 ## volume
                    if prodwater >= (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol : prodwater = (1-ratio_proppant) * (1-ratio_flowback) * tot_inj_vol
                    prod_vol_per_frac = prodwater / nfrac_select                    
    
                    for j in range(nfrac_select):
                        
                        center_ = np.array([x_loc_fracture[j],0,well_depth])
                        x = cart_grid_rotate[:,0] - center_[0]
                        y = cart_grid_rotate[:,1] - center_[1] 
                        z = cart_grid_rotate[:,2] - center_[2]
                        sol_ = prod_vol_per_frac * self.pore_pressure_undrained_DC(S,etha,diffu,x,y,z,k,l)
                        output_[:,i] -= sol_

        return output_