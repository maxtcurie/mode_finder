import numpy as np
import matplotlib.pyplot as plt

from read_iterdb_file import read_iterdb_file



path='/global/u1/m/maxcurie/max/Cases/jet78697/'
profile_name =path+'jet78697.51005_hager_Z6.0Zeff2.35__negom_alpha1.2_TiTe.iterdb' 		#name of the profile file

#path='/global/u1/m/maxcurie/max/Cases/DIIID175823_250k/'
#profile_name =path+'DIIID175823.iterdb' 		#name of the profile file

rho0, Te, Ti, ne, ni, nz, vrot = read_iterdb_file(profile_name)

'''
de = np.genfromtxt('profiles_e')
di = np.genfromtxt('profiles_i')
dz = np.genfromtxt('profiles_z')

ne = de[:,3]
ni = di[:,3]
nz = dz[:,3]

Te = de[:,2]
Ti = di[:,2]
Tz = dz[:,2]
'''

Z = float(input("Enter Z for impurity:\n"))

#zeff = (ni+nz*Z**2.)*(1./ne)

zeff = (ni+nz*Z**2.)*(1./ni)

tau = zeff*Te/Ti

id = ni/ne

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

rho = float(input("Enter location of interest:\n"))

index=np.argmin(abs(rho-rho0))

print('zeff(x/r='+str(rho)+')='+str(zeff[index]))
print('id(x/r='+str(rho)+')='+str(id[index]))
print('tau(x/r='+str(rho)+')='+str(tau[index]))
