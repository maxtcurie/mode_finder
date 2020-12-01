# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 00:58:16 2020

@author: jlara
"""

import numpy as np

#The following function computes the growth rate of the slab MTM. The input are the physical parameters and it outputs the growth rate in units of omega_{*n}
#Parameters for function:
#nu: normalized to omega_{*n} 
#shat=L_n/L_s
#beta=plasma beta
#eta=L_n/L_T
#ky normalized to rho_i (ion gyroradius)

# function for dicretizing the MTM differential equations
# input domain of the grid (x_max), grid spacing (del_x), order of the desired conductivity desired
# this function calls the function above (??) 
# constructs matrix A using x_grid and del_x. A is defined on (-x_max, x_max)
# this is a matrix representation of the original coupled differential equations



def A_maker(x_max, del_x, w1, v1,eta,alpha,beta,ky,ModIndex,mu,xstar):
    

    # making grid
    x_min = -x_max
    x_grid = np.arange(x_min, x_max+del_x, del_x)
    num = len(x_grid)
    
    # initializing matrix A
    A = np.zeros((2*num-4, 2*num-4), dtype=complex)
    
    # Calling the conductivity function which defines the conductvity order that will be used. 
    # The code also converts conversion of the data types
    k_grid = (2*x_grid * alpha * np.sqrt(1836))/v1 
    #L_maker(leg, lag) ######### calling a function
    w_hat = w1/v1
    
    ### (a bunch of code that would've gone here was commented out in the Mathematica notebook by Joel
    ### so I'm skipping it for now)
    
    
    L11_grid = ((k_grid**2)*(-4.0/117.0 + 38.0j*w_hat/49. + 25.*w_hat**2./59.)+w_hat*(4.0j/49. + 172.*w_hat/517. - 21.0j*(w_hat**2)/53. - 44.0*w_hat**3/337.0))/((k_grid**4)*(1.-118.0j*w_hat/101.0)+(k_grid**2)*(42.0/169. - 158.0j*w_hat/85. - 40.*(w_hat**2)/13.0 + 33.0j*(w_hat**3)/26.)  +  w_hat*(-17.0j/273. - 22.0*w_hat/49.0 + 53.0j*(w_hat**2)/50.0 + 65.0*(w_hat**3)/69.0 - 22.0j*(w_hat**4)/83.0))

    L12grid = ((k_grid**2)*(14./163. - 17.0j*w_hat/47. - 11.*w_hat**2/80.0)+w_hat*(5.0j/88. + 21.0*w_hat/130. - 4.0j*(w_hat**2)/49. - w_hat**3/178.))/(k_grid**4*(1.0 - 118.0j*w_hat/101.)  +  k_grid**2*(42./169. - 13.0j*w_hat/7. - 40.0*w_hat**2/13. + 33.0j*w_hat**3/26.)  +  w_hat*(-17.0j/273. - 13.*w_hat/29. + 18.0j*w_hat**2/17. + 16.0*w_hat**3/17. - 9.0j*w_hat**4/34.))
    if ModIndex==0:
        ModG=1
        
    elif ModIndex==1:
        ModG=np.exp(-((x_grid-mu)/xstar)**2)
    else:
        print("ModIndex must be 0 or 1")
        ModG=0
    
    sigma_grid = ((w1-(1.0+eta))*np.multiply(L11_grid,ModG) - eta*np.multiply(L12grid,ModG))/v1
    #print(sigma_grid)
    #print(ModG)
    # computing the diagonal components of the matrix
    a11 = ky**2 + 2j*1836*beta*sigma_grid
    a12 = -4j*1836**1.5*alpha*beta*sigma_grid*x_grid
    a21 = 4j*alpha*np.sqrt(1836)*sigma_grid/(w1*(w1+1))*x_grid
    a22 = ky**2 - 8j*alpha**2*1836*sigma_grid/(w1*(w1+1))*x_grid**2
    # populating the matrix with the components of the matrix
    # this loop populates the off-diagonal components coming from the finite difference
    for i in range(num-3):
        A[i, i+1], A[i+1, i], A[num-2+i, num-2+i+1],  A[num-2+i+1, num-2+i] \
        = -1/del_x**2, -1/del_x**2, -1/del_x**2, -1/del_x**2

      # this loop populates the diagonal components of the matrix
      ##### testing
    for i in range(num-2):
        A[i,i] = 2/del_x**2 + a11[i+1]
        A[num-2+i, num-2+i] = 2/del_x**2 + a22[i+1]
        A[num-2+i, i] = a21[i+1]
        A[i, num-2+i] = a12[i+1]

    A[0,0] = A[0,0] - 1/del_x**2*(1-ky*del_x)
    A[num-3,num-3] = A[num-3,num-3] - 1/del_x**2*(1-ky*del_x)
    A[num-2,num-2] = A[num-2,num-2] - 1/del_x**2*(1-ky*del_x)
    A[2*num-5,2*num-5] =  A[2*num-5,2*num-5] - 1/del_x**2*(1-ky*del_x)
    return A

def w_finder(x_max, del_x, w_guess, v,ne,alpha,beta,ky,ModIndex,mu,xstar):

    w_minus=w_guess
    # call A_maker to create and populate matrix A
    A = A_maker(x_max, del_x, w_guess, v,ne,alpha,beta,ky,ModIndex,mu,xstar) ##### maybe relabel this as A-minus

    # first step is chosen to be del_w = 0.01 (??? should this be a parameter?)
    del_w = 0.01
    det_A_minus = np.linalg.slogdet(A)
    w0 = w_minus + del_w

    # iterative loop that runs until the correction to the root is very small
    # secant method??
    while np.abs(del_w) > 10**-8:
        A = A_maker(x_max, del_x, w0, v,ne,alpha,beta,ky,ModIndex,mu,xstar)
        det_A0 = np.linalg.slogdet(A)
        del_w = -del_w/(1-(det_A_minus[0]/det_A0[0])*np.exp(det_A_minus[1]-det_A0[1]))
        w_plus = w0 + del_w
        w_minus = w0
        w0 = w_plus
        det_A_minus = det_A0
 
    return w0


def Dispersion(nu,eta,shat,beta,ky,ModIndex,mu,xstar):
  #Fit Parameters
  mu=abs(mu)
  xsigma=1/shat*np.sqrt(1./1836)
  xmax=xsigma*25
  delx=xsigma/50

  w0=w_finder(xmax,delx,1+eta,nu,eta,shat,beta,ky,ModIndex,mu,xstar)
  print("****************")
  print("****************")
  print("****************")
  print("****************")
  print("nu,eta,shat,beta,ky,ModIndex,mu,xstar")
  print(nu,eta,shat,beta,ky,ModIndex,mu,xstar)
  print("omeag+j*gamma")
  print(w0)
  print("****************")
  print("****************")
  print("****************")
  print("****************")

  return w0



#A_maker(10,2,2.,4.0,1.0,0.05,0.005,0.05,1,0.,5.)  
#Dispersion(4.0,1.0,0.05,0.005,0.05,1,2.,5.)

Dispersion(2.6679768409327966, 1.657153606386159, 0.5838248521300481, 0.001952052212782288, 0.011599065071912457, 1, 11., 13.043155807022632)


