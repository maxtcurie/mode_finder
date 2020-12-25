from finite_differences import *
import matplotlib.pyplot as plt
import numpy as np
from interp import *
import math
import csv
from max_pedestal_finder import find_pedestal_from_data
from max_pedestal_finder import find_pedestal
from read_profiles import read_profile_file
from read_profiles import read_geom_file
from read_EFIT_file import get_geom_pars

#Created by Max T. Curie  11/02/2020
#Last edited by Max Curie 11/03/2020
#Supported by scripts in IFS

#location for testing:/global/cscratch1/sd/maxcurie/DIIID_175823/global_scan/n0_20
#**************Block for user*****************************************
#**************Setting up*********************************************
profile_type="ITERDB"           # "ITERDB" "pfile" 
geomfile_type="gfile"          # "gfile"  "GENE_tracor"

path='/global/u1/m/maxcurie/max/Cases/jet78697/'
profile_name = path+'jet78697.51005_hager_Z6.0Zeff2.35__negom_alpha1.2_TiTe.iterdb' 		#name of the profile file
                                            #DIIID175823.iterdb
                                            #p000000
geomfile_name = path+'jet78697.51005_hager.eqdsk'
gene_geomfile_name = 'gene.dat'
gene_geomfile_name2 = 'gene_0001_qmult0.958_hager_78697_nx0320_nz060'
#geomfile_name = 'g162940.02944_670'             #name of the magnetic geometry file
                                            #g000000
                                            #tracer_efit.dat

suffix='dat'                   #The suffix if one choose to use GENE_tracor for q profile
                                #0001, 1, dat

plt.clf()
xgrid, q , R_ref= read_geom_file("gfile",geomfile_name,suffix)
plt.plot(xgrid, q*0.959,label="gfile")
xgrid, q, Lref, R_ref, Bref, x0_from_para = read_geom_file("GENE_tracor",gene_geomfile_name,suffix)
plt.plot(xgrid, q,label="GENE_tracor")
#xgrid, q, Lref, R_ref, Bref, x0_from_para = read_geom_file("GENE_tracor",gene_geomfile_name2,suffix)
#plt.plot(xgrid, q,label="GENE_tracor2")
plt.legend()
plt.show()