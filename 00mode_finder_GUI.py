# -*- coding: utf-8 -*-
"""
Created on 06/19/2021
Updated on 06/23/2021

@author: maxcurie
"""
import tkinter as tk
from tkinter import filedialog 
import numpy as np

from MTMDispersion_tools import Parameter_reader
from MTMDispersion_tools import omega_gaussian_fit

root=tk.Tk()
root.title('Mode finder')
root.iconbitmap('./Physics_helper_logo.ico')

#global varible
global geomfile_name
geomfile_name=''
global profile_name
profile_name=''
global Run_mode
Run_mode=1


#*************Input file*****************
Input_frame=tk.LabelFrame(root, text='Input files',padx=5,pady=5)
Input_frame.grid(row=0,column=0)

#**********Profile file setup****************************
p_frame=tk.LabelFrame(Input_frame, text='Profile file',padx=5,pady=5)
p_frame.grid(row=0,column=0)
tk.Label(p_frame,text='Profile file type').grid(row=0,column=0)

#add the dropdown menu
profile_type_var=tk.StringVar()
profile_type_var.set('Choose type')
p_Options=['ITERDB','pfile','profile_e','profile_both']
p_drop=tk.OptionMenu(p_frame, profile_type_var, *p_Options)
p_drop.grid(row=0,column=1)


p_Path=tk.Label(p_frame,text='Profile file path')
p_Path.grid(row=1,column=0)

def p_Click():
	#find the file
	global p_frame
	global profile_name
	profile_name=filedialog.askopenfilename(initialdir='./',\
										title='select a file', \
										filetypes=( 
											('all files', '*'),\
											('ITERDB','.ITERDB' or '.iterdb'),\
											('pfile','p*'),\
											('profile','profile*'),\
											) \
										)
	global profile_name_box
	profile_name_box=tk.Entry(p_frame, width=50)

	if len(profile_name)>30:
		profile_name_box.insert(0,'...'+profile_name[-30:])
	else:
		profile_name_box.insert(0,profile_name)
	profile_name_box.grid(row=2,column=0)
	

#creat button
p_Path_Button=tk.Button(p_frame, text='Browse for Profile file path',\
					command=p_Click,\
					padx=50, pady=10)

p_Path_Button.grid(row=3,column=0)

#**********Profile file setup****************************



#**********Geometry file setup****************************
g_frame=tk.LabelFrame(Input_frame, text='Geometry file',padx=5,pady=5)
g_frame.grid(row=1,column=0)
tk.Label(g_frame,text='Geometry file type').grid(row=0,column=0)

#add the dropdown menu
geomfile_type_var=tk.StringVar()
geomfile_type_var.set('Choose type')
g_Options=['gfile','GENE_tracer']
g_drop=tk.OptionMenu(g_frame, geomfile_type_var, *g_Options)
g_drop.grid(row=0,column=1)


#find the file
g_Path=tk.Label(g_frame,text='Geometry file path')
g_Path.grid(row=1,column=0)

def g_Click():
	#find the file
	global g_frame
	global geomfile_name
	geomfile_name=filedialog.askopenfilename(initialdir='./',\
									title='select a file', \
									filetypes=( 
										('all files', '*'),
										('gfile/efit','g*')\
										) \
									)
	global geomfile_name_box
	geomfile_name_box=tk.Entry(g_frame,width=50)

	
	if len(geomfile_name)>30:
		geomfile_name_box.insert(0,'...'+geomfile_name[-30:])
	else:
		geomfile_name_box.insert(0,geomfile_name)
	
	geomfile_name_box.grid(row=2,column=0)

#creat button
g_Path_Button=tk.Button(g_frame, text='Browse for Geometry file path',\
					command=g_Click,\
					padx=50, pady=10)

g_Path_Button.grid(row=3,column=0)

#**********Geometry file setup****************************
#*************Input file*****************


#*************Setting********************
Setting_frame=tk.LabelFrame(root, text='Setting',padx=5,pady=5)
Setting_frame.grid(row=1,column=0)

#**********omega percent*********************
tk.Label(Setting_frame,text='Omega* top percentage').grid(row=0,column=0)

omega_percent_inputbox=tk.Entry(Setting_frame, width=20)
omega_percent_inputbox.insert(0,'10.')
omega_percent_inputbox.grid(row=0,column=1)
tk.Label(Setting_frame,text='%').grid(row=0,column=2)
#**********omega percent*********************


#***********************run mode****************

opt_var1= tk.IntVar() #Integar Varible, Other options: StringVar()
opt_var1.set(1)       #Set the default option as option1

def click_mode(a):
	global Run_mode
	Run_mode=a

frame1=tk.LabelFrame(Setting_frame, text='Running Mode Selection',padx=50,pady=40)
frame1.grid(row=1,column=0)

option_button11=tk.Radiobutton(frame1, text='Rational surface alignment(Fast)',\
							variable=opt_var1, value=1,\
							command=lambda: click_mode(opt_var1.get()))
option_button11.grid(row=1,column=0)

option_button12=tk.Radiobutton(frame1, text='Global Disperion Calculation',\
							variable=opt_var1, value=2,\
							command=lambda: click_mode(opt_var1.get()))
option_button12.grid(row=2,column=0)

option_button13=tk.Radiobutton(frame1, text='Local Disperion Calculation',\
							variable=opt_var1, value=3,\
							command=lambda: click_mode(opt_var1.get()))
option_button13.grid(row=3,column=0)

#***********************run mode****************




#*************Setting********************



#*******************Show all the setting and load the data********************
omega_percent=float(omega_percent_inputbox.get())
profile_type=profile_type_var.get()
geomfile_type=geomfile_type_var.get()

q_scale=1.
manual_ped=-1
manual_zeff=-1
Z=6.
suffix='.dat'

def Load_data(profile_name,geomfile_name,\
			q_scale,manual_ped,manual_zeff):
	
	profile_type=profile_type_var.get()
	geomfile_type=geomfile_type_var.get()

	print('omega_percent='+str(omega_percent)+'%')
	print('Run_mode='+str(Run_mode))
	print('geomfile_name='+str(geomfile_name))
	print('profile_name ='+str(profile_name))
	print('geomfile_type='+str(geomfile_type))
	print('profile_type ='+str(profile_type))


	uni_rhot,nu,eta,shat,beta,ky,q,mtmFreq,\
		omegaDoppler,omega_n,omega_n_GENE,\
		xstar,Lref, R_ref, rhoref=\
			Parameter_reader(profile_type,profile_name,\
                geomfile_type,geomfile_name,\
                q_scale,manual_ped,manual_zeff,suffix,Z=6.,\
                plot=False,output_csv=True)


Print_Setting_Button=tk.Button(root, text='Load the data',\
					command=lambda: Load_data(profile_name,geomfile_name,\
									q_scale,manual_ped,manual_zeff),\
					padx=50, pady=10)

Print_Setting_Button.grid(row=3,column=0)


#*******************Show all the setting and load the data********************

#creat the GUI
root.mainloop()






    
