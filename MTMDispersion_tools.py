# -*- coding: utf-8 -*-
"""
Created on 11/27/2020

@author: maxcurie, jlara
"""
import numpy as np
import pandas as pd
from finite_differences import *
import matplotlib.pyplot as plt
from interp import *
import math
import csv
from scipy import optimize

#For GUI
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)

from max_pedestal_finder import find_pedestal
from max_pedestal_finder import find_pedestal_from_data
from read_profiles import read_profile_file
from read_profiles import read_geom_file
from read_EFIT_file import get_geom_pars
from max_stat_tool import *
#from DispersionRelationDeterminant import Dispersion
from DispersionRelationDeterminantFullConductivityZeff import Dispersion

#The following function computes the growth rate of the slab MTM. The input are the physical parameters and it outputs the growth rate in units of omega_{*n}
#Parameters for function:
#nu: normalized to omega_{*n} 
#shat=L_n/L_s
#beta=plasma beta
#eta=L_n/L_T
#ky normalized to rho_i (ion gyroradius)
#mu is the location, xstar is the spread of the guassian distribution modeling the omega*e

#Late Edited by Max Curie, 12/04/2020

#gamma=0 is the not real mode; gamma=-100, stable and no real mode(for large mu cases)

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2)


def gaussian_fit(x,data):
    x,data=np.array(x), np.array(data)

    #warnings.simplefilter("error", OptimizeWarning)
    judge=0

    try:
        popt, pcov = optimize.curve_fit(gaussian, x,data)  
        max_index=np.argmax(data)
        
        
        plt.clf()
        plt.plot(x,data, label="data")
        plt.plot(x, gaussian(x, *popt), label="fit")
        plt.axvline(x[max_index],color='red',alpha=0.5)
        plt.axvline(x[max_index]+popt[2],color='red',alpha=0.5)
        plt.axvline(x[max_index]-popt[2],color='red',alpha=0.5)
        plt.legend()
        plt.show()

        judge=int(input("Is the fit okay?  0. No 1. Yes"))

    except RuntimeError:
        print("Curve fit failed, need to restrict the range")

        judge=0
        
        popt=[0]*3  
        max_index=np.argmax(data)
        popt[0]=data[max_index] #amplitud
        popt[1]=x[max_index]    #mean

        plt.clf()
        plt.plot(x,data, label="data")
        plt.axvline(x[max_index],color='red',alpha=0.5)
        plt.legend()
        plt.show()

        while judge==0:

            popt[2]=float(input("sigma="))*np.sqrt(2.)  #stddev

            plt.clf()
            plt.plot(x,data, label="data")
            plt.plot(x, gaussian(x, *popt), label="fit")
            plt.axvline(x[max_index],color='red',alpha=0.5)
            plt.axvline(x[max_index]+popt[2],color='red',alpha=0.5)
            plt.axvline(x[max_index]-popt[2],color='red',alpha=0.5)
            plt.legend()
            plt.show()

            judge=int(input("Is the fit okay?  0. No 1. Yes"))


    while judge==0:

        popt=[0]*3  
        max_index=np.argmax(data)
        popt[0]=data[max_index] #amplitud
        popt[1]=x[max_index]    #mean

        popt[2]=float(input("sigma="))*np.sqrt(2.)  #stddev

        plt.clf()
        plt.plot(x,data, label="data")
        plt.plot(x, gaussian(x, *popt), label="fit")
        plt.axvline(x[max_index],color='red',alpha=0.5)
        plt.axvline(x[max_index]+popt[2],color='red',alpha=0.5)
        plt.axvline(x[max_index]-popt[2],color='red',alpha=0.5)
        plt.legend()
        plt.show()

        judge=int(input("Is the fit okay?  0. No 1. Yes"))

            
    amplitude=popt[0]
    mean     =popt[1]
    stddev   =popt[2] 

    return amplitude,mean,stddev

def gaussian_fit_GUI(root,x,data):
    x,data=np.array(x), np.array(data)

    global amplitude
    global mean
    global stddev


    top=tk.Toplevel(root)
    top.title('Gaussian Fit')
    #root.geometry("500x500")
    
    #load the icon for the GUI 
    top.iconbitmap('./Physics_helper_logo.ico')

    #warnings.simplefilter("error", OptimizeWarning)

    #top=tk.LabelFrame(root, text='Gaussian Fit',padx=20,pady=20)
    #top.grid(row=0,column=1)

    frame_plot=tk.LabelFrame(top, text='Plot of the data and fitting',padx=20,pady=20)
    frame_plot.grid(row=0,column=0)
    fig = Figure(figsize = (5, 5),
                dpi = 100)
    plot1 = fig.add_subplot(111)
    plot1.plot(x,data, label="data")

    try:
        popt, pcov = optimize.curve_fit(gaussian, x,data)  
        max_index=np.argmax(data)

        amplitude=popt[0]
        mean=popt[1]
        stddev=popt[2]

        plot1.plot(x, gaussian(x, *popt), label="fit")
        plot1.axvline(mean,color='red',alpha=0.5)
        plot1.axvline(mean+stddev,color='red',alpha=0.5)
        plot1.axvline(mean-stddev,color='red',alpha=0.5)
        plot1.legend()

    except RuntimeError:
        print("Curve fit failed, need to fit manually")
        pass  

    canvas = FigureCanvasTkAgg(fig,master = frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas,frame_plot)
    toolbar.update()
    canvas.get_tk_widget().pack()

    root1=tk.LabelFrame(top, text='User Box',padx=20,pady=20)
    root1.grid(row=0,column=1)


    frame1=tk.LabelFrame(root1, text='Accept/Reject the fit',padx=20,pady=20)
    frame1.grid(row=0,column=0)
    #import an option button
    opt_var1= tk.IntVar() #Integar Varible, Other options: StringVar()

    option_button11=tk.Radiobutton(frame1, text='Accept the fit',\
                                variable=opt_var1, value=1,\
                                command=lambda: click_button(opt_var1.get(), top,root1))
    option_button11.grid(row=1,column=0)

    option_button12=tk.Radiobutton(frame1, text='Enter manually',\
                                variable=opt_var1, value=2,\
                                command=lambda: click_button(opt_var1.get(), top,root1))
    option_button12.grid(row=2,column=0)
        

    
    def save_exit(amplitude,mean,stddev):
        looping=0
        top.quit()

    def click_button(value, top, root1):

        global amplitude
        global mean
        global stddev 

        label_list=[]

        frame_data=tk.LabelFrame(root1, text='Current fitting parameter',padx=80,pady=20)
        frame_data.grid(row=1,column=0)
        label_list.append(frame_data)

        amplitude_string=f'amplitude = {amplitude}'
        mean_string     =f'mean      = {mean}'
        std_string      =f'stddev    = {stddev}'
        amplitude_string=amplitude_string+' '*(50-len(amplitude_string))
        mean_string=mean_string+' '*(50-len(mean_string))
        std_string=std_string+' '*(50-len(std_string))

        amp_label =tk.Label(frame_data,text=amplitude_string)
        mean_label=tk.Label(frame_data,text=mean_string)
        std_label =tk.Label(frame_data,text=std_string)
        amp_label.grid(row=0,column=0)
        mean_label.grid(row=1,column=0)
        std_label.grid(row=2,column=0)
        label_list.append(amp_label)
        label_list.append(mean_label)
        label_list.append(std_label)

        def plot_manual_fit(x,data,mean_Input,sigma_Input,top,label_list):
            #frame_plot.grid_forget()

            frame_plot=tk.LabelFrame(top, text='Plot of the data and fitting',padx=20,pady=20)
            frame_plot.grid(row=0,column=0)
            fig = Figure(figsize = (5, 5),
                        dpi = 100)
            plot1 = fig.add_subplot(111)
            plot1.plot(x,data, label="data")
            
            global amplitude
            global mean
            global stddev 

            mean=float(mean_Input)
            amplitude=data[np.argmin(abs(x-mean))]
            stddev=float(sigma_Input)


            frame_data=label_list[0]
            amp_label=label_list[1]
            mean_label=label_list[2]
            std_label=label_list[3]

            amp_label.grid_forget()
            mean_label.grid_forget()
            std_label.grid_forget()

            amplitude_string=f'amplitude = {amplitude}'
            mean_string     =f'mean      = {mean}'
            std_string      =f'stddev    = {stddev}'

            amplitude_string=amplitude_string+' '*(50-len(amplitude_string))
            mean_string=mean_string+' '*(50-len(mean_string))
            std_string=std_string+' '*(50-len(std_string))


            amp_label =tk.Label(frame_data,text=amplitude_string)
            mean_label=tk.Label(frame_data,text=mean_string)
            std_label =tk.Label(frame_data,text=std_string)
            amp_label.grid(row=0,column=0)
            mean_label.grid(row=1,column=0)
            std_label.grid(row=2,column=0)

            plot1.plot(x, gaussian(x, amplitude, mean, stddev), label="fit")
            plot1.axvline(mean,color='red',alpha=0.5)
            plot1.axvline(mean+stddev,color='red',alpha=0.5)
            plot1.axvline(mean-stddev,color='red',alpha=0.5)
            plot1.legend()
        
            canvas = FigureCanvasTkAgg(fig,master = frame_plot)
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas,frame_plot)
            toolbar.update()
            canvas.get_tk_widget().pack()

        

        if value==1:
            #'Accept the fit'
            myButton2=tk.Button(root1, text='Save and Continue', \
                command=lambda : save_exit(amplitude,mean,stddev))
            myButton2.grid(row=3,column=0)

        elif value==2:
            #Label1.grid_forget()
            #'Enter manually'
            frame_input=tk.LabelFrame(root1, text='Manual Input',padx=10,pady=10)
            frame_input.grid(row=2,column=0)
            tk.Label(frame_input,text='mu(center) = ').grid(row=0,column=0)
            mean_Input_box=tk.Entry(frame_input, width=30, bg='green', fg='white')
            mean_Input_box.insert(0,'')
            mean_Input_box.grid(row=0,column=1)
        
            tk.Label(frame_input,text='sigma(spread) = ').grid(row=1,column=0)
            sigma_Input_box=tk.Entry(frame_input, width=30, bg='green', fg='white')
            sigma_Input_box.insert(0,'')
            sigma_Input_box.grid(row=1,column=1)

            

            Plot_Button=tk.Button(frame_input, text='Plot the Manual Fit',\
                command=lambda: plot_manual_fit(x,data,\
                    float( mean_Input_box.get()  ),\
                    float( sigma_Input_box.get() ),\
                    top,label_list\
                    )  \
                )
            Plot_Button.grid(row=2,column=1)

            Save_Button=tk.Button(root1, text='Save and Continue',\
                     state=tk.DISABLED)#state: tk.DISABLED, or tk.NORMAL
            Save_Button.grid(row=3,column=0)

    stddev=abs(stddev)
    print(f'amplitude,mean,stddev={amplitude},{mean},{stddev}')
    top.mainloop()

    
    return amplitude,mean,stddev



def omega_gaussian_fit(root,x,data,rhoref,Lref,show=False):
    amplitude,mean,stddev=gaussian_fit_GUI(root,x,data)
    print(f'amplitude,mean,stddev={amplitude},{mean},{stddev}')
    mean_rho=mean*Lref/rhoref         #normalized to rhoi
    xstar=abs(stddev*Lref/rhoref)
    
    popt=[0]*3
    popt[0] = amplitude
    popt[1] = mean     
    popt[2] = stddev   

    print(popt)
    print(mean_rho,xstar)

    if show==True:
        plt.clf()
        plt.plot(x,data, label="data")
        plt.plot(x, gaussian(x, *popt), label="fit")
        plt.legend()
        plt.show()

    return mean_rho,xstar

 

#return nu,ky for the case n_tor=1 for the given location(default to be pedestal)
def Parameter_reader(profile_type,profile_name,\
    geomfile_type,geomfile_name,Run_mode,\
    q_scale,manual_ped,manual_zeff,suffix,root,Z=6.,\
    plot=False,output_csv=True):
    n0=1.
    mref = 2.        # mass of ion in proton mass

    if profile_type=="ITERDB":
        rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0 = read_profile_file(profile_type,profile_name,geomfile_name,suffix)
    elif profile_type=="pfile":
        rhot0, rhop0, te0, ti0, ne0, ni0, nz0, vrot0 = read_profile_file(profile_type,profile_name,geomfile_name,suffix)


    if geomfile_type=="gfile": 
        xgrid, q, R_ref= read_geom_file(geomfile_type,geomfile_name,suffix)
    elif geomfile_type=="GENE_tracer":
        xgrid, q, Lref, R_ref, Bref, x0_from_para = read_geom_file(geomfile_type,geomfile_name,suffix)

    q=q*q_scale

    if geomfile_type=="GENE_tracer" and profile_type!="profile":
        rhot0_range_min=np.argmin(abs(rhot0-xgrid[0]))
        rhot0_range_max=np.argmin(abs(rhot0-xgrid[-1]))
        rhot0=rhot0[rhot0_range_min:rhot0_range_max]
        rhop0=rhop0[rhot0_range_min:rhot0_range_max]
        te0=te0[rhot0_range_min:rhot0_range_max]
        ti0=ti0[rhot0_range_min:rhot0_range_max]
        ne0=ne0[rhot0_range_min:rhot0_range_max]
        ni0=ni0[rhot0_range_min:rhot0_range_max]
        nz0=nz0[rhot0_range_min:rhot0_range_max]
        vrot0=vrot0[rhot0_range_min:rhot0_range_max]

    uni_rhot = np.linspace(min(rhot0),max(rhot0),len(rhot0)*10)

    te_u = interp(rhot0,te0,uni_rhot)
    ne_u = interp(rhot0,ne0,uni_rhot)
    ni_u = interp(rhot0,ni0,uni_rhot)
    print(str((len(rhot0),len(nz0),len(uni_rhot))))
    nz_u = interp(rhot0,nz0,uni_rhot)
    vrot_u = interp(rhot0,vrot0,uni_rhot)
    q      = interp(xgrid,q,uni_rhot)
    tprime_e = -fd_d1_o4(te_u,uni_rhot)/te_u
    nprime_e = -fd_d1_o4(ne_u,uni_rhot)/ne_u
    qprime = fd_d1_o4(q,uni_rhot)/q


    #center_index = np.argmax((tprime_e*te_u+nprime_e*ne_u)[0:int(len(tprime_e)*0.99)])
    
    if int(manual_ped)==-1: 
        if geomfile_type=="gfile": 
            midped, topped=find_pedestal(file_name=geomfile_name, path_name='', plot=False)
        elif geomfile_type=="GENE_tracer":
            midped=x0_from_para
        x0_center = midped
    else:
        x0_center=manual_ped


    print('mid pedestal is at r/a = '+str(x0_center))
    if geomfile_type=="gfile": 
        Lref, Bref, R_major, q0, shat0=get_geom_pars(geomfile_name,x0_center)
    

    print("Lref="+str(Lref))
    print("x0_center="+str(x0_center))

    index_begin=np.argmin(abs(uni_rhot-x0_center+2*(1.-x0_center)))

    te_u = te_u[index_begin:len(uni_rhot)-1]
    ne_u = ne_u[index_begin:len(uni_rhot)-1]
    ni_u = ni_u[index_begin:len(uni_rhot)-1]
    nz_u = nz_u[index_begin:len(uni_rhot)-1]
    vrot_u = vrot_u[index_begin:len(uni_rhot)-1]
    q      = q[index_begin:len(uni_rhot)-1]
    tprime_e = tprime_e[index_begin:len(uni_rhot)-1]
    nprime_e = nprime_e[index_begin:len(uni_rhot)-1]
    qprime   = qprime[index_begin:len(uni_rhot)-1]
    uni_rhot = uni_rhot[index_begin:len(uni_rhot)-1]

    Lt=1./tprime_e
    Ln=1./nprime_e
    
    center_index = np.argmin(abs(uni_rhot-x0_center))

    q0      = q[center_index]

    ne=ne_u/(10.**19.)      # in 10^19 /m^3
    ni=ni_u/(10.**19.)      # in 10^19 /m^3
    nz=nz_u/(10.**19.)      # in 10^19 /m^3
    te=te_u/1000.          #in keV
    m_SI = mref *1.6726*10**(-27)
    me_SI = 9.11*10**(-31)
    c  = 1.
    qref = 1.6*10**(-19)
    #refes to GENE manual
    coll_c=2.3031*10**(-5)*Lref*ne/(te)**2*(24-np.log(np.sqrt(ne*10**13)/(te*1000)))
    coll_ei=4.*coll_c*np.sqrt(te*1000.*qref/me_SI)/Lref
    nuei=coll_ei
    beta=403.*10**(-5)*ne*te/Bref**2.

    nref=ne_u[center_index]
    te_mid=te_u[center_index]
    Tref=te_u[center_index] * qref

    cref = np.sqrt(Tref / m_SI)
    Omegaref = qref * Bref / m_SI / c
    rhoref = cref / Omegaref 
    rhoref_temp = rhoref * np.sqrt(te_u/te_mid) 
    kymin=n0*q0*rhoref/(Lref*x0_center)
    kyGENE =kymin * (q/q0) * np.sqrt(te_u/te_mid) * (x0_center/uni_rhot) #Add the effect of the q varying
    #from mtm_doppler
    omMTM = kyGENE*(tprime_e+nprime_e)
    gyroFreq = 9.79E3/np.sqrt(mref)*np.sqrt(te_u)/Lref
    mtmFreq = omMTM*gyroFreq/(2.*np.pi*1000.)
    omegaDoppler = abs(vrot_u*n0/(2.*np.pi*1000.))
    omega=mtmFreq + omegaDoppler

    

    if int(manual_zeff)==-1:
        zeff = ( (ni+Z**2*nz)/ne )[center_index]
    else:
        zeff=manual_zeff

    print('zeff='+str(zeff))

    omega_n_GENE=kyGENE*(nprime_e)       #in cs/a
    omega_n=omega_n_GENE*gyroFreq/(2.*np.pi*1000.)  #in kHz

    #coll_ei=coll_ei/(1000.)  #in kHz
    coll_ei=coll_ei/(2.*np.pi*1000.)  #in kHz

    Lq=1./(Lref/(R_ref*q)*qprime)


    shat=Ln/Lq
    eta=Ln/Lt
    ky=kyGENE*np.sqrt(2.)
    nu=(coll_ei)/(np.max(omega_n))



    nuei=nu*omega_n_GENE/omega_n

    

    if plot==True:
        plt.clf()
        #plt.title('mode number finder')
        plt.xlabel('r/a')
        plt.ylabel('omega*, kHz') 
        plt.plot(uni_rhot,mtmFreq,label='omega*p')
        plt.plot(uni_rhot,omega_n,label='omega*n')
        plt.axvline(uni_rhot[np.argmax(mtmFreq)],color='red',alpha=1.,label='peak of omega*p')
        plt.axvline(uni_rhot[np.argmax(omega_n)],color='green',alpha=1.,label='peak of omega*n')
        plt.plot(uni_rhot,rhoref_temp,color='purple',label='rhoref')
        plt.legend()
        plt.show()

        print("rho i for peak of omega*p: "+str(rhoref_temp[np.argmax(mtmFreq)]))
        print("rho i for peak of omega*n: "+str(rhoref_temp[np.argmax(omega_n)]))

        plt.clf()
        plt.xlabel('r/a')
        plt.ylabel('eta') 
        plt.plot(uni_rhot,eta,label='eta')
        plt.show()

        plt.clf()
        #plt.title('mode number finder')
        plt.xlabel('r/a')
        plt.ylabel('omega*(Lab), kHz') 
        plt.plot(uni_rhot,omega,label='omega*(Lab)')
        plt.show()

        plt.clf()
        plt.xlabel('r/a')
        plt.ylabel('eta') 
        plt.plot(uni_rhot,eta,label='eta')
        plt.show()

        plt.clf()
        plt.xlabel('r/a')
        plt.ylabel('shat') 
        plt.plot(uni_rhot,shat,label='shat')
        plt.show()


        plt.clf()
        plt.xlabel('r/a')
        plt.ylabel('beta') 
        plt.plot(uni_rhot,beta,label='beta')
        plt.show()
        
        plt.clf()
        plt.xlabel('r/a')
        plt.ylabel('ky rhoi') 
        plt.plot(uni_rhot,ky,label='ky')
        plt.show()

    if Run_mode==2 :              #calculate global disparsion
        mean_rho,xstar=omega_gaussian_fit(root,uni_rhot,mtmFreq,rhoref*np.sqrt(2.),Lref,plot)
    else :
        mean_rho,xstar=0,0
    
    #if abs(mean_rho) + abs(xstar)<0.0001:
    #    print('abs(mean_rho) + abs(xstar)<0.0001')
    #    pass() 

    if output_csv==True:
        with open('profile_output.csv','w') as csvfile:
            data = csv.writer(csvfile, delimiter=',')
            data.writerow(['x/a','nu_ei(kHz)','omega*n(kHz)','omega* plasma(kHz)','Doppler shift(kHz)','nu/omega*n','eta','shat','beta','ky rhoi(for n=1)'])
            for i in range(len(uni_rhot)):
                data.writerow([uni_rhot[i],coll_ei[i],omega_n[i],mtmFreq[i],omegaDoppler[i],nu[i],eta[i],shat[i],beta[i],ky[i]])
        csvfile.close()
    
    return uni_rhot,nu,eta,shat,beta,ky,q,mtmFreq,omegaDoppler,omega_n,\
            omega_n_GENE,xstar,Lref, R_ref, rhoref, zeff

#scan the Dispersion for the given location(default to be pedestal)
def Dispersion_list(uni_rhot,nu,eta,shat,beta,ky,ModIndex,mu,xstar,plot):
    nx=len(uni_rhot)
    #print(nx)
    gamma_complex_temp=0
    #factor_temp=0
    gamma_complex=[]
    #factor=[]

    for i in range(nx):
        gamma_complex_temp=Dispersion(nu[i],eta[i],shat[i],beta[i],ky[i],ModIndex,mu[i],xstar) 
        gamma_complex.append(gamma_complex_temp)
        #factor.append(factor_temp)

        #gamma_complex_temp,factor_temp=Dispersion(nu[i],eta[i],shat[i],beta[i],ky[i],ModIndex,mu,xstar) 
        #gamma_complex.append(gamma_complex_temp)
        #factor.append(factor_temp)
    
    gamma_complex=np.asarray(gamma_complex)
    #factor=np.asarray(factor)
    gamma=gamma_complex.imag
    omega=gamma_complex.real

    if plot==True:
        plt.clf()
        plt.xlabel('r/a')
        plt.ylabel('gamma/omega*n') 
        plt.plot(uni_rhot,gamma,label='gamma')
        plt.show()

    return gamma,omega
    #return gamma,omega,factor


#this function takes the q and n, returns the locations of the rational surfaces
def Rational_surface(uni_rhot,q,n0):
    x_list=[]
    m_list=[]
    qmin = np.min(q)
    qmax = np.max(q)

    #print("q_min is: "+str(qmin))
    #print("q_max is: "+str(qmax))
    m_min = math.ceil(qmin*n0)
    m_max = math.floor(qmax*n0)
    mnums = np.arange(m_min,m_max+1)

    print(m_min,m_max)
    #or m in range(m_min,m_max+1):
    for m in mnums:
    	#print(m)
        q0=float(m)/float(n0)
        index0=np.argmin(abs(q-q0))
        if abs(q[index0]-q0)<0.1:
            x_list.append(uni_rhot[index0])
            m_list.append(m)

    #print(x_list)
    #x_list=np.asarray(x_list)
    #m_list=np.asarray(m_list)
    return x_list, m_list

#this function finds the a peak of the 
def Peak_of_drive(uni_rhot,mtmFreq,omegaDoppler,omega_percent):
    x_peak_range=[]
    x_range_ind=[]
    #omega=mtmFreq+omegaDoppler
    
    omega=mtmFreq
    
    #if manual_ped==1:
    #    mid_ped0
    #mid_ped0=0.958
    omega_max=np.max(omega)
    print(mtmFreq)
    print(omega_max)
    omega_min=omega_max*(100.-omega_percent)/100.
    for i in range(len(uni_rhot)):
        if omega[i] >= omega_min:
            x_peak_range.append(uni_rhot[i])
            x_range_ind.append(i)
    print("x="+str(min(x_peak_range))+"~"+str(max(x_peak_range)))
    return x_peak_range, x_range_ind

#scan toroidial mode number
def Dispersion_n_scan(uni_rhot,nu,eta,shat,beta,ky,q,omega_n,\
    omega_n_GENE,mtmFreq,omegaDoppler,x_peak_range,x_range_ind,\
    n_min,n_max,rhoi,Lref,Run_mode,xstar,Zeff,plot,output_csv):
    ind_min  =min(x_range_ind)
    ind_max  =max(x_range_ind)
    uni_rhot_full=uni_rhot
    nu_full=nu
    eta_full=eta
    shat_full=shat
    beta_full=beta
    ky_full=ky
    q_full=q
    omega_n_full=omega_n
    omega_n_GENE_full=omega_n_GENE
    omegaDoppler_full=omegaDoppler
    mtmFreq_full=mtmFreq

    uni_rhot_top=uni_rhot[ind_min:ind_max]
    print("uni_rhot="+str(min(uni_rhot_top))+"~"+str(max(uni_rhot_top)))
    nu_top=nu[ind_min:ind_max]
    eta_top=eta[ind_min:ind_max]
    shat_top=shat[ind_min:ind_max]
    beta_top=beta[ind_min:ind_max]
    ky_top=ky[ind_min:ind_max]
    q_top=q[ind_min:ind_max]
    omega_n_top=omega_n[ind_min:ind_max]
    omega_n_GENE_top=omega_n_GENE[ind_min:ind_max]
    omegaDoppler_top=omegaDoppler[ind_min:ind_max]
    mtmFreq_top=mtmFreq[ind_min:ind_max]
    omega_star_max=max(mtmFreq_top)
    x_peak=uni_rhot_top[np.argmax(mtmFreq_top)]

    print("x_peak="+str(x_peak))

    n_list=[]
    m_list=[]
    x_list=[]
    gamma_list=[]
    omega_list=[]
    #factor_list=[]
    gamma_list_kHz=[]
    omega_list_kHz=[]
    omega_list_Lab_kHz=[]
    omega_star_list_kHz=[]
    omega_star_list_Lab_kHz=[]
    nu_ei_kHz_list=[]
    omega_omega_peak_list=[]
    kHz_to_cs_a_list=[]

    nu_list=[]
    eta_list=[]
    shat_list=[]
    beta_list=[]
    ky_list=[]
    mu_list=[]
    xstar_list=[]
    
    if plot==True:
        plt.clf()
        plt.title('mode number scan')
        plt.xlabel('r/a')
        plt.ylabel('gamma/omega*n') 
        
        

    for n0 in range(n_min,n_max+1):
        print("************n="+str(n0)+"************")

        #gamma,omega=Dispersion_list(uni_rhot_full,nu_full/float(n0),eta_full,shat_full,beta_full,ky_full*float(n0),ModIndex,mu,xstar,plot=False)
        
        #gamma,omega,factor=Dispersion_list(uni_rhot_full,nu_full/float(n0),eta_full,shat_full,beta_full,ky_full*float(n0),ModIndex,mu,xstar,plot=False)
        x0_list, m0_list=Rational_surface(uni_rhot_top,q_top,n0)
        #print(x_list)
        

        for i in range(len(x0_list)):
            x=x0_list[i]

            #print(x)
            m=m0_list[i]
            #print(m)

            x_index=np.argmin(abs(x-uni_rhot_top))
            
            mu=(x-x_peak)*Lref/rhoi

            if Run_mode==1:
                gamma=1.
                omega=0.
            elif Run_mode==2:#Global 
                ModIndex=1
                gamma=Dispersion(nu_top[x_index]/float(n0),Zeff,eta_top[x_index],\
                    shat_top[x_index],beta_top[x_index],ky_top[x_index]*float(n0),ModIndex,mu,xstar)
                gamma_complex=gamma
                gamma=gamma_complex.imag
                omega=gamma_complex.real
            elif Run_mode==3:#Global
                ModIndex=0
                gamma=Dispersion(nu_top[x_index]/float(n0),Zeff,eta_top[x_index],\
                    shat_top[x_index],beta_top[x_index],ky_top[x_index]*float(n0),ModIndex,mu,xstar)
                gamma_complex=gamma
                gamma=gamma_complex.imag
                omega=gamma_complex.real

            if plot==True and gamma>0:
                plt.plot(x,gamma)   #,label='n='+str(n0))

            
            kHz_to_cs_a=omega_n_GENE_top[x_index]/omega_n_top[x_index]   # kHz * (kHz_to_cs_a)= cs/a unit
            omega_n_temp=omega_n_top[x_index]*float(n0)
            gamma_kHz=gamma*omega_n_temp
            omega_kHz=omega*omega_n_temp
            omega_Lab_kHz=omega_kHz-omegaDoppler_top[x_index]*float(n0)
            omega_star_kHz=mtmFreq_top[x_index]*float(n0)
            omega_star_Lab_kHz=mtmFreq_top[x_index]*float(n0)+omegaDoppler_top[x_index]*float(n0)
            nu_ei_kHz=( nu_top[x_index] * omega_n_top[x_index] )
            #nu_ei/omega*e in plasma frame

            gamma_list_kHz.append(gamma_kHz)
            kHz_to_cs_a_list.append(kHz_to_cs_a)
            omega_list_kHz.append(omega_kHz)
            omega_list_Lab_kHz.append(omega_Lab_kHz)
            omega_star_list_kHz.append(omega_star_kHz)
            omega_star_list_Lab_kHz.append(omega_star_Lab_kHz)
            nu_ei_kHz_list.append(nu_ei_kHz)
            gamma_list.append(gamma)
            omega_list.append(omega)
            #factor_list.append(factor)
            n_list.append(n0)
            m_list.append(m)
            x_list.append(x)
            omega_omega_peak_list.append(omega_star_kHz/omega_star_max/float(n0))

            nu_list.append(nu_top[x_index]/float(n0))
            eta_list.append(eta_top[x_index])
            shat_list.append(shat_top[x_index])
            beta_list.append(beta_top[x_index])
            ky_list.append(ky_top[x_index]*float(n0))
            mu_list.append(mu)
            xstar_list.append(xstar)

            


            #print("x="+str(x)+", gamma(kHz)="+str(gamma_kHz))
            if plot==True and gamma>0:
                plt.axvline(x,color='red',alpha=0.5)

    if plot==True:
        plt.axvline(-100,color='red',alpha=1,label='unstable MTM')
        plt.axvline(min(uni_rhot_top),color='green',label="near the peak of omega*")
        plt.axvline(max(uni_rhot_top),color='green',label="near the peak of omega*")
        plt.xlim(min(uni_rhot_top),max(uni_rhot_top))
        plt.plot(uni_rhot_top,mtmFreq_top,label='omega*',color='blue')
        plt.plot(uni_rhot_top,q_top,label='safety factor',color='purple')
        plt.grid()
        plt.legend()
        plt.show()
    
    #print(len(x_list),len(n_list),len(m_list))
    if output_csv==True:
        with open('MTM_dispersion_n_scan.csv','w') as csvfile:
            data = csv.writer(csvfile, delimiter=',')
            data.writerow([   'x/a',     'n',      'm',      'gamma(kHz)',      'gamma(cs/a)',    'nu_ei(kHz)','omega*/omega*_max' ,'omega_plasma(kHz)','omega_lab(kHz)',   'omega_star_plasma(kHz)','omega_star_lab(kHz)','nu', 'eta', 'shat', 'beta', 'ky', 'mu', 'xstar'])
            for i in range(len(x_list)):
                data.writerow([x_list[i],n_list[i],m_list[i],gamma_list_kHz[i],gamma_list_kHz[i]*kHz_to_cs_a_list[i],nu_ei_kHz_list[i],omega_omega_peak_list[i],omega_list_kHz[i],omega_list_Lab_kHz[i],omega_star_list_kHz[i],omega_star_list_Lab_kHz[i],nu_list[i], eta_list[i], shat_list[i], beta_list[i], ky_list[i], mu_list[i], xstar_list[i]])
        csvfile.close()

    return x_list,n_list,m_list,gamma_list,omega_list,gamma_list_kHz,omega_list_kHz,omega_list_Lab_kHz,omega_star_list_kHz,omega_star_list_Lab_kHz

#define a normal distribusion, input the x axis as x_list, output norm(x_list)
def normal(x_list,x0,mu,sigma):
    #return x0*1./(sigma*sqrt(2.*np.pi))*np.exp(-1./2.*((x-mu)/sigma)**2.)
    return x0*np.exp(-1./2.*((x_list-mu)/sigma)**2.)

#Calculate the gamma as function of frequency
#input the omega*n for n=1  
def Spectrogram(gamma_list_kHz,omega_list_kHz):

    omega_min=min(omega_list_kHz)
    omega_max=max(omega_list_kHz)
    #print(omega_min)
    #print(omega_max)
    f=np.arange(omega_min,omega_max,0.01)
    gamma_f=np.zeros(len(np.arange(omega_min,omega_max,0.01)))
    #print(gamma_f)
    for i in range(len(gamma_list_kHz)):
        if gamma_list_kHz[i]>0:
            x0=gamma_list_kHz[i]
        else:
            x0=0
        mu=omega_list_kHz[i]
        sigma=0.2
        gamma_f=gamma_f+normal(f,x0,mu,sigma)
    return f,gamma_f

def Spectrogram_2_frames(gamma_list_kHz,omega_list_kHz,omega_list_Lab_kHz,bins,plot):
    
    f_lab,gamma_f_lab=Spectrogram(gamma_list_kHz,omega_list_Lab_kHz)
    #f_lab_pd_frame=pd.DataFrame(f_lab, columns=['a']) 
    #f_lab_smooth=f_lab_pd_frame.diff().rolling(window=dt).mean()
    #gamma_f_lab_smooth=gamma_f_lab.diff().rolling(window=dt).mean()
    if plot==True:
        plt.clf()
        plt.title('Spectrogram in lab frame')
        plt.xlabel('f(kHz)')
        plt.ylabel('gamma(a.u.)') 
        plt.plot(f_lab,gamma_f_lab)
        plt.show()

        plt.clf()
        plt.title('Spectrogram in lab frame(rolling average)')
        plt.xlabel('f(kHz)')
        plt.ylabel('gamma(a.u.)') 
        plt.plot(smooth(f_lab,bins)[0],smooth(gamma_f_lab,bins)[0])
        plt.show()


    f_plasma,gamma_f_plasma=Spectrogram(gamma_list_kHz,omega_list_kHz)
    if plot==True:
        plt.clf()
        plt.title('Spectrogram in plasma frame')
        plt.xlabel('f(kHz)')
        plt.ylabel('gamma(a.u.)') 
        plt.plot(f_plasma,gamma_f_plasma)
        plt.show()

        plt.clf()
        plt.title('Spectrogram in plasma frame(rolling average)')
        plt.xlabel('f(kHz)')
        plt.ylabel('gamma(a.u.)') 
        plt.plot(smooth(f_plasma,bins)[0],smooth(gamma_f_plasma,bins)[0])
        plt.show()
    return f_lab,gamma_f_lab,f_plasma,gamma_f_plasma

def peak_scan(uni_rhot,nu,eta,shat,beta,ky,q,omega_n,omega_n_GENE,mtmFreq,omegaDoppler,n0,Lref,rhoi,ModIndex,xstar,plot_peak_scan,csv_peak_scan):
    center_index=np.argmax(mtmFreq)
    x0=uni_rhot[center_index]
    if choose_location==True:
        x0=location
        center_index=np.argmin(abs(uni_rhot-x0))
    x0_center=uni_rhot[center_index]
    #x0=0.98
    #center_index=np.argmin(abs(uni_rhot-x0))
    #print(x0)
    nu0=nu[center_index]/float(n0)
    eta0=eta[center_index]
    shat0=shat[center_index]
    beta0=beta[center_index]
    ky0=ky[center_index]*float(n0)

    omega_n_0=omega_n[center_index]*float(n0)
    omega_n_GENE_0=omega_n_GENE[center_index]*float(n0)
    mtmFreq0=mtmFreq[center_index]*float(n0)
    omegaDoppler0=omegaDoppler[center_index]*float(n0)
    
    uni_rhot_list=[]
    nu_list=[]
    eta_list=[]
    shat_list=[]
    beta_list=[]
    ky_list=[]
    nu_list_omega_star=[]

    #print(uni_rhot[center_index])
    nmax=400
    

    nu_min=nu0*(1.-nu_percent/2.)
    d_nu=nu0*nu_percent/float(nmax)

    for i in range(nmax):
        nu_list.append(nu_min+float(i)*d_nu)
        uni_rhot_list.append(x0)
        eta_list.append(eta0)
        shat_list.append(shat0)
        beta_list.append(beta0)
        ky_list.append(ky0)
        nu_list_omega_star.append((nu_min+float(i)*d_nu)*omega_n_0/(mtmFreq0))
        mu_list.append((x0_center-x0)*Lref/rhoi)
    
    gamma,omega=Dispersion_list(uni_rhot_list,nu_list,eta_list,shat_list,beta_list,ky_list,ModIndex,mu_list,xstar,plot=False)
    #gamma,omega,factor=Dispersion_list(uni_rhot_list,nu_list,eta_list,shat_list,beta_list,ky_list,ModIndex,mu,xstar,plot=False)
    gamma_cs_a=gamma*omega_n_GENE_0
    
    if csv_peak_scan==True:
        d = {'nu_omegan':nu_list,'eta':eta_list,'shat':shat_list,'beta':beta_list,'ky':ky_list,'nu_ei_omega_plasma':nu_list_omega_star,'gamma_cs_a':gamma_cs_a}
        df=pd.DataFrame(d, columns=['nu_omegan','eta','shat','beta','ky','nu_ei_omega_plasma','gamma_cs_a'])
        df.to_csv('nu_scan.csv',index=False)
    #print(nu_list)
    #nu_list_omega_star=nu_list*omega_n_0/(mtmFreq0+omegaDoppler0)

    if plot_peak_scan==True:
        plt.clf()
        plt.title('Nu scan')
        plt.xlabel('nu/omega*')
        plt.ylabel('gamma(cs/a)') 
        plt.plot(nu_list_omega_star,gamma_cs_a)
        plt.show()


def MTM_scan(profile_name,geomfile_name,q_scale,omega_percent,bins,n_min,n_max,plot_profile,plot_n_scan,csv_profile,csv_n_scan): 
    #return nu,ky for the case n_tor=1 for the given location(default to be pedestal)
    uni_rhot,nu,eta,shat,beta,ky,q,mtmFreq,omegaDoppler,omega_n,omega_n_GENE,xstar,Lref,rhoref=Parameter_reader(profile_name,geomfile_name,q_scale,manual_ped,mid_ped0,plot=plot_profile,output_csv=csv_profile)
    print(mtmFreq)
    print(omegaDoppler)
    x_peak_range, x_range_ind=Peak_of_drive(uni_rhot,mtmFreq,omegaDoppler,omega_percent)

    #scan toroidial mode number
    x_list,n_list,m_list,gamma_list,\
    omega_list,gamma_list_kHz,\
    omega_list_kHz,omega_list_Lab_kHz,\
    omega_star_list_kHz,omega_star_list_Lab_kHz\
    =Dispersion_n_scan(uni_rhot,nu,eta,shat,beta,ky,q,\
    omega_n,omega_n_GENE,mtmFreq,omegaDoppler,x_peak_range,x_range_ind,\
    n_min,n_max,rhoref,Lref,ModIndex,xstar,plot=plot_n_scan,output_csv=csv_n_scan)

    '''
    x_list,n_list,m_list,gamma_list,\
    omega_list,factor_list,gamma_list_kHz,\
    omega_list_kHz,omega_list_Lab_kHz,\
    omega_star_list_kHz,omega_star_list_Lab_kHz\
    =Dispersion_n_scan(uni_rhot,nu,eta,shat,beta,ky,q,\
    omega_n,omega_n_GENE,mtmFreq,omegaDoppler,x_peak_range,x_range_ind,\
    n_min,n_max,ModIndex,mu,xstar,plot=plot_n_scan,output_csv=csv_n_scan)
    '''

    f_lab,gamma_f_lab,f_plasma,gamma_f_plasma=Spectrogram_2_frames(gamma_list_kHz,omega_star_list_kHz,omega_star_list_Lab_kHz,bins,plot=plot_spectrogram)
    return x_list,n_list,m_list,gamma_list,omega_list


def coll_scan(profile_name,geomfile_name,q_scale,n0,plot_profile,plot_peak_scan,csv_profile,csv_peak_scan): 
    uni_rhot,nu,eta,shat,beta,ky,q,mtmFreq,omegaDoppler,omega_n,omega_n_GENE,xstar,Lref,rhoi=Parameter_reader(profile_name,geomfile_name,q_scale,manual_ped,mid_ped0,plot=plot_profile,output_csv=csv_profile)
    peak_scan(uni_rhot,nu,eta,shat,beta,ky,q,omega_n,omega_n_GENE,mtmFreq,omegaDoppler,n0,Lref,rhoi,ModIndex,xstar,plot_peak_scan,csv_peak_scan)

