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
profile_type="pfile"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"         # "gfile"  "GENE_tracor"

#profile_name_list=['p169509.03069_416','p169509.04218_465']
#geomfile_name_list=['g169509.03069_416','g169509.04218_465']
#name_list=['t=3069ms','t=4218ms']

profile_name_list=['p169510.01966_108','p169510.03090_595','p169510.04069_173']
geomfile_name_list=['g169510.01966_108','g169510.03090_595','g169510.04069_173']
name_list=['t=1966ms','t=3090ms','t=4069ms']


#path='/global/u1/m/maxcurie/max/Cases/DIIID162940_Ehab/'
#path=''
#profile_name = 'DIIID175823.iterdb'
#profile_name = 'p169509.03069_416'	#1
#profile_name = 'p169509.04218_465'	#2
#profile_name = 'p169510.01966_108'	#3
#profile_name = 'p169510.03090_595'	#4
#profile_name = 'p169510.04069_173'	#5
#profile_name = path+'DIIID162940.iterdb'
#profile_name =path+'jet78697.51005_hager_Z6.0Zeff2.35__negom_alpha1.2_TiTe.iterdb' 		#name of the profile file
                                            #DIIID175823.iterdb
                                            #p000000
#geomfile_name = 'g175823.04108_257x257'
#geomfile_name = path+'jet78697.51005_hager.eqdsk'
#geomfile_name = 'tracer_efit.dat'
#geomfile_name = 'g169509.03069_416'	#1
#geomfile_name = 'g169509.04218_465'	#2
#geomfile_name = 'g169510.01966_108'	#3
#geomfile_name = 'g169510.03090_595'	#4
#geomfile_name = 'g169510.04069_173'	#5

#plt.clf()
x_list=[]
q_list=[]
om_list=[]
for i in range(len(profile_name_list)):
    profile_name=profile_name_list[i]
    geomfile_name=geomfile_name_list[i]


    #geomfile_name = 'gene.dat'      #name of the magnetic geometry file
                                            #g000000
                                            #tracer_efit.dat

    suffix='dat'                   #The suffix if one choose to use GENE_tracor for q profile
                                #0001, 1, dat

    f_max=200.     #upper bound of the frequency experimentally observed 
    f_min=0        #lower bound of the frequency experimentally observed 
    plot = 1         #set to 1 if you want to plot the result
    report=1         #set to 1 if you want to export a csv report
    omega_percent=5.  #choose the omega within the top that percent defined in(0,100)
    n0_min=1         #minmum mode number (include) that finder will cover
    n0_max=20      #maximum mode number (include) that finder will cover
    q_scale= 1.       #set the q to q*q_scale
    mref = 2.        # mass of ion in proton mass, D=2.  ,T=3. 

    x0_center_choose=1  #change to 1 if one wants to choose mid-pedestal manually 
    x0_center_pick=0.95
    #**************End of Setting up*********************************************
    #**************End of Block for user******************************************



    #*************Loading the data******************************************
    if profile_type=="ITERDB":
        rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0 = read_profile_file(profile_type,profile_name,geomfile_name,suffix)
    else:
        rhot0, rhop0, te0, ti0, ne0, ni0, vrot0 = read_profile_file(profile_type,profile_name,geomfile_name,suffix)


    if geomfile_type=="gfile": 
        xgrid, q, R_ref= read_geom_file(geomfile_type,geomfile_name,suffix)
    elif geomfile_type=="GENE_tracor":
        xgrid, q, Lref, R_ref, Bref, x0_from_para = read_geom_file(geomfile_type,geomfile_name,suffix)

    q=q*q_scale

    if geomfile_type=="GENE_tracor" and profile_type!="profile":
        rhot0_range_min=np.argmin(abs(rhot0-xgrid[0]))
        rhot0_range_max=np.argmin(abs(rhot0-xgrid[-1]))
        rhot0=rhot0[rhot0_range_min:rhot0_range_max]
        rhop0=rhop0[rhot0_range_min:rhot0_range_max]
        te0=te0[rhot0_range_min:rhot0_range_max]
        ti0=ti0[rhot0_range_min:rhot0_range_max]
        ne0=ne0[rhot0_range_min:rhot0_range_max]
        ni0=ni0[rhot0_range_min:rhot0_range_max]
        vrot0=vrot0[rhot0_range_min:rhot0_range_max]


    uni_rhot = np.linspace(min(rhot0),max(rhot0),len(rhot0)*10.)

    te_u = interp(rhot0,te0,uni_rhot)
    ne_u = interp(rhot0,ne0,uni_rhot)
    vrot_u = interp(rhot0,vrot0,uni_rhot)
    q      = interp(xgrid,q,uni_rhot)
    tprime_e = -fd_d1_o4(te_u,uni_rhot)/te_u
    nprime_e = -fd_d1_o4(ne_u,uni_rhot)/ne_u


    #midped, topped=find_pedestal(file_name=geomfile_name, path_name='', plot=False)
    #(q/q0) * np.sqrt(te_u/te_mid) * (x0_center/uni_rhot)



    if x0_center_choose==1: 
        x0_center=x0_center_pick
    else:
        if geomfile_type=="gfile": 
            midped, topped=find_pedestal(file_name=geomfile_name, path_name='', plot=False)
        elif geomfile_type=="GENE_tracor":
            midped=x0_from_para
        x0_center = midped

    print('mid pedestal is at r/a = '+str(x0_center))

    if geomfile_type=="gfile": 
        Lref, Bref, R_major, q0, shat0=get_geom_pars(geomfile_name,x0_center)

    print("Lref="+str(Lref))
    print("x0_center="+str(x0_center))

    index_begin=np.argmin(abs(uni_rhot-x0_center+2*(1.-x0_center)))

    te_u = te_u[index_begin:len(uni_rhot)-1]
    ne_u = ne_u[index_begin:len(uni_rhot)-1]
    vrot_u = vrot_u[index_begin:len(uni_rhot)-1]
    q      = q[index_begin:len(uni_rhot)-1]
    tprime_e = tprime_e[index_begin:len(uni_rhot)-1]
    nprime_e = nprime_e[index_begin:len(uni_rhot)-1]
    uni_rhot = uni_rhot[index_begin:len(uni_rhot)-1]

    center_index = np.argmin(abs(uni_rhot-x0_center))

    
    #*************End of loading the data******************************************

    #****************Start setting up ******************

    q0      = q[center_index]
    ne = ne_u[center_index]
    te = te_u[center_index] #it is in eV
    #Bref=float(Bref_Gauss)/10000
    m_SI = mref *1.6726*10**(-27)
    c  = 1
    qref = 1.6*10**(-19)
    nref = ne
    Tref = te * qref
    cref = np.sqrt(Tref / m_SI)
    Omegaref = qref * Bref / m_SI / c
    rhoref = cref / Omegaref 

    n0=1.
    m0 = n0*q0
    ky=n0*q0*rhoref/(Lref*x0_center)
    kymin = ky
    print("kymin="+str(kymin))
    n0_global = n0
    te_mid = te_u[center_index]
    kyGENE =kymin * (q/q0) * np.sqrt(te_u/te_mid) * (x0_center/uni_rhot) #Add the effect of the q varying
    #***Calculate omeage star********************************
    #from mtm_doppler
    omMTM = kyGENE*(tprime_e+nprime_e)
    gyroFreq = 9.79E3/np.sqrt(mref)*np.sqrt(te_u)/Lref
    #print("gyroFreq="+str(gyroFreq[center_index]))
    mtmFreq0 = omMTM*gyroFreq/2./np.pi/1000.
    omegaDoppler0 = abs(vrot_u*n0_global/2./np.pi/1E3)
    omega0 = mtmFreq0 + omegaDoppler0
    #***End of Calculate omeage star**************************
    x_list.append(uni_rhot)
    q_list.append(q)
    om_list.append(mtmFreq0)


plt.clf()
for i in range(len(x_list)):
    plt.plot(x_list[i],om_list[i],label=name_list[i])
plt.xlabel('r/a')
plt.ylabel('Frequency(kHz)') 
#plt.xlim(0.9,1)
#plt.ylim(0,max(mtmFreq + omegaDoppler))
plt.legend()
plt.show()

plt.clf()
for i in range(len(x_list)):
    plt.plot(x_list[i],q_list[i],label=name_list[i])
plt.xlabel('r/a')
plt.ylabel('safety factor') 
#plt.xlim(0.9,1)
#plt.ylim(0,max(mtmFreq + omegaDoppler))
plt.legend()
plt.show()


