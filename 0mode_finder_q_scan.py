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
profile_type="profile"           # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="GENE_tracor"          # "gfile"  "GENE_tracor"

path='/global/u1/m/maxcurie/max/Cases/jet78697/'
profile_name = path+'jet78697.51005_hager_Z6.0Zeff2.35__negom_alpha1.2_TiTe.iterdb' 		#name of the profile file
                                            #DIIID175823.iterdb
                                            #p000000
geomfile_name = 'gene.dat'
#geomfile_name = 'g162940.02944_670'             #name of the magnetic geometry file
                                            #g000000
                                            #tracer_efit.dat

suffix='dat'                   #The suffix if one choose to use GENE_tracor for q profile
                                #0001, 1, dat
q_scale_list=np.arange(0.94,0.96,0.0005)

f_max=500      #upper bound of the frequency experimentally observed 
f_min=0        #lower bound of the frequency experimentally observed 
report=1         #set to 1 if you want to export a csv report
omega_percent=20.  #choose the omega within the top that percent defined in(0,100)
n0_min=1         #minmum mode number (include) that finder will cover
n0_max=13      #maximum mode number (include) that finder will cover

mref = 2.        # mass of ion in proton mass, D=2.  ,T=3. 

x0_center_choose=0  #change to 1 if one wants to choose mid-pedestal manually 
x0_center_pick=0.98
#**************End of Setting up*********************************************
#**************End of Block for user******************************************




#*************Loading the data******************************************

plot = 0         #set to 1 if you want to plot the result

rhot0, rhop0, te0, ti0, ne0, ni0, vrot0 = read_profile_file(profile_type,profile_name,geomfile_name,suffix)
if geomfile_type=="gfile": 
    xgrid, q = read_geom_file(geomfile_type,geomfile_name,suffix)
elif geomfile_type=="GENE_tracor":
    xgrid, q, Lref, Bref, x0_from_para = read_geom_file(geomfile_type,geomfile_name,suffix)


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

q_1      = q[index_begin:len(uni_rhot)-1]
te_u = te_u[index_begin:len(uni_rhot)-1]
ne_u = ne_u[index_begin:len(uni_rhot)-1]
vrot_u = vrot_u[index_begin:len(uni_rhot)-1]
tprime_e = tprime_e[index_begin:len(uni_rhot)-1]
nprime_e = nprime_e[index_begin:len(uni_rhot)-1]
uni_rhot = uni_rhot[index_begin:len(uni_rhot)-1]


center_index = np.argmin(abs(uni_rhot-x0_center))
q0_1      = q_1[center_index]
ne = ne_u[center_index]
te = te_u[center_index]


#*************End of loading the data******************************************

if report==1:
    with open('0mode_number_finder_report_q_scale.csv','w') as csvfile:
        data = csv.writer(csvfile, delimiter=',')
        data.writerow(['q_scale','n ','m ','ky(' + str(x0_center)+')','ky(r)   ','frequency(kHz)           ','location(r/a)            ','omega(cs/a)    ','Drive(omega*/omega*_max)'])
        csvfile.close()
print(str(q_scale_list))
for q_scale in q_scale_list:        #set the q to q*q_scale
    #****************Start setting up ******************
    print("q_scale="+str(q_scale))
    q=q_1*q_scale
    q0=q0_1*q_scale
    #it is in eV
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




    #***Find the range where the omega is within the top omega_percent%
    omega_range=[]
    range_ind=[]
    omega_max=np.max(mtmFreq0)
    omega_min=omega_max*(100.-omega_percent)/100.
    for i in range(len(uni_rhot)):
        if mtmFreq0[i] >= omega_min:
            omega_range.append(uni_rhot[i])
            range_ind.append(i)
    range_min=min(omega_range)
    ind_min  =min(range_ind)
    range_max=max(omega_range)
    ind_max  =max(range_ind)

    #***End of Find the range where the omega is within the top omega_percent%

    print('n=1, max omega is ' + str(omega_max) +'kHz')

    #******************End setting up ****************

    #****************Start scanning mode number*************
    kymin_range=[]
    ky_range=[]
    n0_range=[]
    m0_range=[]
    f_range=[]
    f_GENE_range=[]
    x_range=[]
    drive_range=[]

    for n0 in range(n0_min,n0_max+1):
        n0_TEMP=0
    
    #***********Calculating the ky********************************
        #From SI_Gauss_GENE_unit.py
        m0 = q0*n0
        ky=kymin*n0
        omega = omega0*n0
        mtmFreq =  mtmFreq0*n0
        omegaDoppler = omegaDoppler0*n0
        if plot==1:
            plt.clf()
            plt.title('mode number finder')
            plt.xlabel('r/a')
            plt.ylabel('frequency (kHz)') 
            plt.plot(uni_rhot,mtmFreq + omegaDoppler,label='Diamagnetic plus Doppler (MTM in lab frame)')
            plt.plot(uni_rhot,mtmFreq ,label='Diamagnetic plus Doppler (MTM in plasma frame)')


    #*******Find the range within the frequency range************************
        range_ind2=[]
        for i in range(len(uni_rhot)):
            if omega[i] >= f_min and omega[i] <= f_max:
                range_ind2.append(i)
    #*******Find the range within the frequency range************************
    

    #find the rational surfaces

    #From the plot_mode_structures.py
        qmin = np.min(q[ind_min:ind_max])
        qmax = np.max(q[ind_min:ind_max])
        mmin = math.ceil(qmin*n0)
        mmax = math.floor(qmax*n0)
        mnums = np.arange(mmin,mmax+1)
        qrats = mnums/float(n0)

        if ky <= 5:
    #***********End of Calculating the ky************************
            for i in range(len(qrats)):
                n = int(n0)
                m = int(mmin + i)
                ix = np.argmin(abs(q-qrats[i])) 
                if uni_rhot[ix] >= range_min and uni_rhot[ix] <= range_max:
                    if ix in range_ind2:
                        print('ky='+str(ky))
                        print('(m,n)='+str((m,n)))
                        temp_str=str((n,m))
                        if plot==1:
                            plt.axvline(uni_rhot[ix],color='red', label= temp_str)
                        n0_TEMP=n0_TEMP+1
                        kymin_range.append(ky)
                        ky_range.append(kyGENE[ix])
                        n0_range.append(n)
                        m0_range.append(m)
                        f_range.append(omega[ix])
                        f_GENE_range.append(omega[ix]*2*np.pi*1000/gyroFreq[ix])
                        x_range.append(uni_rhot[ix])
                        drive_range.append(mtmFreq[ix]/(float(n)*omega_max))

                    else:
                        if plot==1:
                            plt.axvline(uni_rhot[ix],color='green')
                else:
                    if plot==1:
                        plt.axvline(uni_rhot[ix],color='green')
        else:
            for i in range(len(qrats)):
                n = int(n0)
                m = int(mmin + i)
                ix = np.argmin(abs(q-qrats[i]))
                if plot==1:
                    plt.axvline(uni_rhot[ix],color='green')

        if n0_TEMP > 0 and plot==1:
            plt.xlim(0.9,1)
            plt.ylim(0,max(mtmFreq + omegaDoppler))
            plt.legend()
            plt.savefig('mode_number_finder_n0='+str(n0)+'.png')
    print('**********************************************')
    print('**************Start of report*****************')
    if len(ky_range)==0:
        print('There is no unstabel MTM')
    else:
        print('ky range from '+str(min(ky_range))+' to '+str(max(ky_range)))
        print('n0 range from '+str(min(n0_range))+' to '+str(max(n0_range)))
    print('**********************************************')
    print('***************End of report******************')

    if report==1:
        with open('0mode_number_finder_report_q_scale.csv','a+') as csvfile:
            data = csv.writer(csvfile, delimiter=',')
            for i in range(len(n0_range)):
                data.writerow([q_scale,n0_range[i],m0_range[i],kymin_range[i],ky_range[i],f_range[i],x_range[i],f_GENE_range[i],drive_range[i]])
        csvfile.close()

   