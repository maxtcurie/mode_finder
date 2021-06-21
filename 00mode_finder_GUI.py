# -*- coding: utf-8 -*-
"""
Created on 06/19/2021

@author: maxcurie
"""
import tkinter as tk
from tkinter import filedialog 


root=tk.Tk()
root.title('Mode finder')
root.iconbitmap('./Physics_helper_logo.ico')


Input_frame=tk.LabelFrame(root, text='Input files',padx=5,pady=5)
Input_frame.grid(row=0,column=0)

#**********Profile file setup****************************
p_frame=tk.LabelFrame(Input_frame, text='Profile file',padx=5,pady=5)
p_frame.grid(row=0,column=0)
tk.Label(p_frame,text='Profile file type').grid(row=0,column=0)

#add the dropdown menu
profile_type=tk.StringVar()
profile_type.set('Choose type')
p_Options=['ITERDB','pfile','profile_e','profile_both']
p_drop=tk.OptionMenu(p_frame, profile_type, *p_Options)
p_drop.grid(row=0,column=1)


p_Path=tk.Label(p_frame,text='Profile file path')
p_Path.grid(row=1,column=0)

def p_Click():
	#find the file
	global p_frame
	p_frame.filename=filedialog.askopenfilename(initialdir='./',\
										title='select a file', \
										filetypes=( 
											('all files', '*'),\
											('ITERDB','.ITERDB' or '.iterdb'),\
											('pfile','p*'),\
											('profile','profile*'),\
											) \
										)
	global profile_name
	profile_name=tk.Entry(p_frame)
	profile_name.insert(0,p_frame.filename)
	profile_name.grid(row=2,column=0)
	

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
geomfile_type=tk.StringVar()
geomfile_type.set('Choose type')
g_Options=['gfile','GENE_tracer']
g_drop=tk.OptionMenu(g_frame, geomfile_type, *g_Options)
g_drop.grid(row=0,column=1)


#find the file
g_Path=tk.Label(g_frame,text='Geometry file path')
g_Path.grid(row=1,column=0)

def g_Click():
	#find the file
	global g_frame
	g_frame.filename=filedialog.askopenfilename(initialdir='./',\
									title='select a file', \
									filetypes=( 
										('all files', '*'),
										('gfile','g.')\
										) \
									)
	global geomfile_name
	geomfile_name=tk.Entry(g_frame)
	geomfile_name.insert(0,g_frame.filename)
	geomfile_name.grid(row=2,column=0)

#creat button
g_Path_Button=tk.Button(g_frame, text='Browse for Geometry file path',\
					command=g_Click,\
					padx=50, pady=10)

g_Path_Button.grid(row=3,column=0)

#**********Geometry file setup****************************

#creat the GUI
root.mainloop()
