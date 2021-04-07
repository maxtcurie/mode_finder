# -*- coding: utf-8 -*-
"""
Created on 01/28/2021

@author: maxcurie
"""

import pandas as pd
import numpy as np
#from mpi4py import MPI

from MPI_tools import task_dis
from DispersionRelationDeterminantFullConductivityZeff import Dispersion

csvfile_name='./Scan_list_template.csv'

data_set=[]

'''
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

print("size:"+str(size))
print("rank:"+str(rank))
'''

def Dispersion_calc(path,filename):
    
    gamma_complex_temp=0
    gamma_complex=[]
    data=pd.read_csv(path+'/'+filename)  	#data is a dataframe. 

    nu=data['nu']
    nx=len(nu)
    zeff=data['zeff'][0]
    eta=data['eta']
    shat=data['shat']
    beta=data['beta']
    ky=data['ky']
    mu=data['mu']
    xstar=data['xstar'][0]
    ModIndex=1
    task_list=[]
    for i in range(nx):
        print('*******nu[i],zeff,eta[i],shat[i],beta[i],ky[i],ModIndex,mu[i],xstar********')
        print(nu[i],zeff,eta[i],shat[i],beta[i],ky[i],ModIndex,mu[i],xstar)
        print('***************')
        gamma_complex.append(1)
        #********This is the part that needs to be paraelled ****************
        task_list.append([nu[i],zeff,eta[i],shat[i],beta[i],ky[i],ModIndex,mu[i],xstar])
        gamma_complex_temp=Dispersion(nu[i],zeff,eta[i],shat[i],beta[i],ky[i],ModIndex,mu[i],xstar) 
        #gamma_complex.append(gamma_complex_temp)
        #********This is the part that needs to be paraelled *****************

    size=6
    task_dis_list=task_dis(size,task_list)
    for i in range(size):
        task_dis_list[i]

    gamma_complex=np.asarray(gamma_complex)
    #factor=np.asarray(factor)
    gamma=gamma_complex.imag
    omega=gamma_complex.real

    #data['gamma(kHz)']=gamma*data['omn(kHz)']
    #data['gamma(cs/a)']=data['gamma(kHz)']*data['kHz_to_cs_a']
    #data['omega_plasma(kHz)']=omega*data['omn(kHz)']
    #data['omega_lab(kHz)']=data['omega_plasma(kHz)']+data['omega_star_lab(kHz)']-data['omega_star_plasma(kHz)']

    data.to_csv(path+'/0_MPI_calc_'+filename,index=False)

    return gamma,omega

data=pd.read_csv(csvfile_name)      #data is a dataframe. 
Path_list=data['Path']
profile_name_list=data['profile_name']

for i in range(len(profile_name_list)):
    filename='MTM_dispersion_n_scan'+profile_name_list[i]+'.csv'
    gamma,omega=Dispersion_calc(Path_list[i],filename)

