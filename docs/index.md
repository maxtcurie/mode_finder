## Welcome to Mode finder GitHub page

**IMPORTANT** The mode finder package is the freely avavible for everyone. It will be appricated if one can **cite** the papers and presentations that are related to this project. 

```markdown
** Larakers, J.,Curie, M.T., Hatch, D. R., Hazeltine, R., & Mahajan, S. M. (2021). Tearing modes in thetokamak pedestal.Physics Review Letters(In review)**

@article{Larakers_prl,
title = {Tearing modes in the tokamak pedestal},
 author = {J.L. Larakers and Curie, M.T. and D. R. Hatch and R.D. Hazeltine and S. M. Mahajan},
 year = {2021},
 journal = {Physics Review Letters(In review)}
}

** Larakers, J., D. R., Hazeltine, R., & Mahajan, S. M. (2020). A comprehensive conductivity model for drift and micro-tearing modes [https://doi.org/10.1063/5.0006215](url)**


@article{Larakers_pop,
title = {Tearing modes in the tokamak pedestal},
 author = {J.L. Larakers and R.D. Hazeltine and S. M. Mahajan},
 year = {2020},
 journal = {Physics of Plasmas},
 url = {https://doi.org/10.1063/5.0006215}
}

#Curie, M.T., Halfmoon, M., Larakers, J., Hassan, E., Chen, J., Brower, D., Hatch, D.,Kotschenreuther, M., Mahajan, S., DIII-D team, & JET team. (2020). Exploring reduced predictivemodels for magnetic fluctuations, APS DPP.

@INPROCEEDINGS{APS2020,
author={\textbf{Curie, M.T.} and M.R. Halfmoon and Joel Larakers and Ehab Hassan and J. Chen and D.L. Brower and D.R. Hatch and M.T. Kotschenreuther and S.M. Mahajan and {DIII-D team} and {JET team}},
year={2020},
title={Exploring reduced predictive models for magnetic fluctuations},
 address = {APS DPP},
 timestamp = {2020.11.05},
keyword={meeting}
}

#
comment
#
```


The mode finder has a few options for the running the scripts: 
1. **0mode_finder_main.py**:   running the mode finder only by alignment of rational surfaces and peak of $\omega_*e$ along
2. **0MTMDispersion.py**:      running the mode finder with the disperison calculatin
3. **0MTMDispersion_MPI.py**:  running the mode finder with the disperison calculatin that is MPI enabled 

One can find the user block in the code in MTMDisperion.py


```markdown
#**************Block for user******************************************
#**************Setting up*********************************************

profile_type="ITERDB"          # "ITERDB" "pfile", "profile_e", "profile_both" 
geomfile_type="gfile"         # "gfile"  "GENE_tracor"

#path='/global/u1/m/maxcurie/max/Cases/DIIID162940_Ehab/'
path='/global/u1/m/maxcurie/max/Cases/jet78697/'
#path=''
profile_name = 'jet78697.51005_hager_Z6.0Zeff2.35__negom_alpha1.2_TiTe.iterdb' 
#profile_name = path+'DIIID162940.iterdb'
#profile_name =path+'jet78697.51005_hager_Z6.0Zeff2.35__negom_alpha1.2_TiTe.iterdb'		#name of the profile file
                                            #DIIID175823.iterdb
                                            #p000000
#geomfile_name = 'g175823.04108_257x257'
geomfile_name = 'jet78697.51005_hager.eqdsk'
#geomfile_name = 'tracer_efit.dat'

#geomfile_name = 'gene_0001_qmult0.958_hager_78697_nx0320_nz060'     #name of the magnetic geometry file
                                            #g000000
                                            #tracer_efit.dat

suffix='dat'            	    #The suffix if one choose to use GENE_tracor for q profile
                                #0001, 1, dat

run_mode_finder=True        #Change to True if one want to run mode finder 
run_nu_scan=False           #Change to True if one want to run collisionality scan 
ModIndex=1 					# 1 is taking global effect, 0 is only local effect 

omega_percent=10.                      #choose the omega within the top that percent defined in(0,100)
#q_scale=1.015
q_scale=1. #0.949 #0.955
n_min=10                                #minmum mode number (include) that finder will cover
n_max=30                              #maximum mode number (include) that finder will cover
bins=800                               #sizes of bins to smooth the function
plot_profile=True                     #Set to True is user want to have the plot of the profile
plot_n_scan=False                      #Set to True is user want to have the plot of the gamma over n
csv_profile=False                    #Set to True is user want to have the csv file "profile_output.csv" of the profile
csv_n_scan=True                       #Set to True is user want to have the csv file "MTM_dispersion_n_scan.csv" of the gamma over n
plot_spectrogram=False
peak_of_plasma_frame=False             #Set to True if one want to look around the peak of omega*e in plasam frame

zeff_manual=False  #2.35	#Effective charges due impurity
Z=6.		#charge of impurity
manual_ped=0
mid_ped0=0.97


#******For scaning********
scan_n0=3.
choose_location=False    #Change to True if one wants to change the location manually 
location=0.984139203080616
plot_peak_scan=True
csv_peak_scan=True
nu_percent=10  #about the nu0 for x% 1=100%
#**************End of Setting up*********************************************
#**************End of Block for user******************************************
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/maxcurie1996/mode_finder/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
