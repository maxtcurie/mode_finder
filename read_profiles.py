import csv
import numpy as np
import math

from max_pedestal_finder import find_pedestal
from read_iterdb_file import read_iterdb_file
from read_pfile import p_to_iterdb_format
from read_EFIT import read_EFIT
from read_write_geometry import read_geometry_global
from parIOWrapper import init_read_parameters_file
#Created by Max T. Curie  11/02/2020
#Last edited by Max Curie 11/02/2020
#Supported by scripts in IFS

def input():
    temp=input("The profile file type: 1. ITERDB  2. pfile")
    suffix="1"
    if temp==1:
        profile_type="ITERDB"    # "ITERDB" "pfile"  "GENE"
    elif temp==2:
        profile_type="pfile"    # "ITERDB" "pfile"  "GENE"
    else:
        print("Please type 1 or 2")

    profile_name = input("The profile file name: ")


    temp=input("The geometry file type: 1. gfile  2. GENE_tracor")
    if temp==1:
        geomfile_type="gfile"    
    elif temp==2:
        geomfile_type="GENE_tracor"  
        suffix=input("Suffix (0001, 1, dat): ")
    else:
        print("Please type 1 or 2")

    geomfile_name = input("The geometry file name: ")   

    return profile_type, geomfile_type, profile_name, geomfile_name, suffix
    

def read_profile_file(profile_type,profile_name,geomfile_name):
    if profile_type=="ITERDB":
        rhot0, te0, ti0, ne0, ni0, nz0, vrot0 = read_iterdb_file(profile_name)
        psi0 = np.linspace(0.,1.,len(rhot0))
        rhop0 = np.sqrt(np.array(psi0))

    elif profile_type=="pfile":
        rhot0, rhop0, te0, ti0, ne0, ni0, vrot0 = p_to_iterdb_format(profile_name,geomfile_name)


    return rhot0, rhop0, te0, ti0, ne0, ni0, vrot0

def read_geom_file(file_type,file_name,suffix="dat"):
    if file_type=="gfile":
        EFITdict = read_EFIT(file_name)
        # even grid of psi_pol, on which all 1D fields are defined
        xgrid = EFITdict['psipn']
        q = EFITdict['qpsi']
        return xgrid, q
        
    elif file_type=="GENE_tracor":
        gpars,geometry = read_geometry_global(file_name)
        q=geometry['q']
        
        if suffix=="dat":
            suffix=".dat"
        else:
            suffix="_"+suffix

        pars = init_read_parameters_file(suffix)
        Lref=pars['Lref']
        Bref=pars['Bref']
        x0_from_para=pars['x0']
        if 'lx_a' in pars:
            xgrid = np.arange(pars['nx0'])/float(pars['nx0']-1)*pars['lx_a']+pars['x0']-pars['lx_a']/2.0
        else:
            xgrid = np.arange(pars['nx0'])/float(pars['nx0']-1)*pars['lx'] - pars['lx']/2.0
        return xgrid, q, Lref, Bref, x0_from_para
    
