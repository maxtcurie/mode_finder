import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from finite_differences import *

T0 = 2.14
KT = 0.00
KTi = 6.92
oT = 0.25
R0 = 2.8

xgrid = np.linspace(0.0,1.0,500)
omt = KT*np.e**(-((xgrid-0.5)/oT)**6)
omti = KTi*np.e**(-((xgrid-0.5)/oT)**6)
Tint = np.empty(len(xgrid))
Tiint = np.empty(len(xgrid))
for i in range(1,len(xgrid)):
    Tint[i] = scipy.integrate.simps(omt[0:i]/R0,xgrid[0:i])
    Tiint[i] = scipy.integrate.simps(omti[0:i]/R0,xgrid[0:i])
T = np.e**(-Tint)
Ti = np.e**(-Tiint)
ix0 = np.argmin(abs(xgrid-0.5))
T = T0/T[ix0]*T
Ti = T0/Ti[ix0]*Ti

Ktest = -R0*fd_d1_o4(np.log(T),xgrid)

plt.plot(xgrid,T,label='Te')
plt.plot(xgrid,Ti,label='Ti')
plt.xlabel('x')
plt.ylabel('T')
plt.legend()
plt.show()

plt.plot(xgrid,Ktest,'x',label='Ktest')
plt.plot(xgrid,omt,label='omt')
plt.plot(xgrid,omti,label='omti')
plt.xlabel('x')
plt.ylabel('omt')
plt.legend()
plt.show()

n0 = 6.483e-2
Kn = 2.22
on = 0.25

omn = Kn*np.e**(-((xgrid-0.5)/on)**6)
nint = np.empty(len(xgrid))
for i in range(1,len(xgrid)):
    nint[i] = scipy.integrate.simps(omn[0:i]/R0,xgrid[0:i])
n = np.e**(-nint)
n = n0/n[ix0]*n

Ktest = -R0*fd_d1_o4(np.log(n),xgrid)

plt.plot(xgrid,n)
plt.xlabel('x')
plt.ylabel('n')
plt.show()

plt.plot(xgrid,Ktest,'x',label='Ktest')
plt.plot(xgrid,omn,label='omn')
plt.xlabel('x')
plt.ylabel('omn')
plt.legend()
plt.show()

f = open('profiles_e_'+str(KT),'w')
f.write('# 1.xgrid 2.xgrid 3.T(kev) 4.n(10^19m^-3)\n')
np.savetxt(f,np.column_stack((xgrid,xgrid,T,n)))
f.close()

f = open('profiles_i_'+str(KTi),'w')
f.write('# 1.xgrid 2.xgrid 3.Ti(kev) 4.n(10^19m^-3)\n')
np.savetxt(f,np.column_stack((xgrid,xgrid,Ti,n)))
f.close()
