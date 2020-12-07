import numpy as np
import matplotlib.pyplot as plt

from read_iterdb_file import read_iterdb_file
from read_EFIT import read_EFIT



path='/global/u1/m/maxcurie/max/Cases/jet78697/'
profile_name =path+'jet78697.51005_hager_Z6.0Zeff2.35__negom_alpha1.2_TiTe.iterdb' 		#name of the profile file

#path='/global/u1/m/maxcurie/max/Cases/DIIID175823_250k/'
#profile_name =path+'DIIID175823.iterdb' 		#name of the profile file

rho1, Te0, Ti0, ne0, ni0, nz0, vrot = read_iterdb_file(profile_name)


EFITdict = read_EFIT(file_name)
        # even grid of psi_pol, on which all 1D fields are defined
xgrid = EFITdict['psipn']
q = EFITdict['qpsi']

rho0 = np.linspace(min(rho1),max(rhot1),len(rhot1)*10.)

te_u = interp(rhot1,Te0,rhot0)
ne_u = interp(rhot0,ne0,uni_rhot)
ni_u = interp(rhot0,ni0,uni_rhot)
vrot_u = interp(rhot0,vrot0,uni_rhot)
q      = interp(xgrid,q,uni_rhot)



x0_center=0.96

center_index = np.argmin(abs(rho0-x0_center))

q0      = q[center_index]

#ne=ne_u/(10.**19.)      # in 10^19 /m^3
#ni=ni_u/(10.**19.)      # in 10^19 /m^3
te=te_u/1000.          #in keV
m_SI = mref *1.6726*10**(-27)
me_SI = 9.11*10**(-31)
c  = 1.
qref = 1.6*10**(-19)
#refes to GENE manual
coll_c=2.3031*10**(-5)*Lref*ne/(te)**2*(24-np.log(np.sqrt(ne*10**13)/(te*1000)))
coll_ei=4*(ni/ne)*coll_c*np.sqrt(te*1000.*qref/me_SI)/Lref
nuei=coll_ei

plt.clf()
plt.plot(rho0,nuei,label='nuei')
plt.legend()
plt.title('nuei')
plt.show()


Z = float(input("Enter Z for impurity:\n"))

#zeff = (ni+nz*Z**2.)*(1./ne)

zeff = (ni+nz*Z**2.)*(1./ni)

tau = zeff*Te/Ti

id = ni/ne


'''
plt.clf()
plt.plot(rho0,ne,label='ne')
plt.plot(rho0,nz,label='nz')
plt.plot(rho0,ni,label='ni')
plt.plot(rho0,zeff,label='zeff')
plt.legend()
plt.title('zeff')
plt.show()

plt.clf()
#plt.plot(rho0,(ni+nz*Z**2.),label='(ni+nz*Z**2.)')
#plt.plot(rho0,1./ne,label='1./ne')
plt.plot(rho0,zeff,label='zeff')
plt.legend()
plt.title('zeff')
plt.show()

plt.plot(rho0,nz)
plt.title('nz')
plt.show()

plt.plot(rho0,id)
plt.title('id')
plt.show()

plt.plot(rho0,tau)
plt.title('tau')
plt.show()
'''
rho = float(input("Enter location of interest:\n"))

index=np.argmin(abs(rho-rho0))

print('zeff(x/r='+str(rho)+')='+str(zeff[index]))
print('id(x/r='+str(rho)+')='+str(id[index]))
print('tau(x/r='+str(rho)+')='+str(tau[index]))
print('nu(x/r='+str(rho)+')='+str(nuei[index])+'cs/a')
