import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpi4py import MPI
from MPI_tools import task_dis

#From: https://www.youtube.com/watch?v=13x90STvKnQ&list=PLQVvvaa0QuDf9IW-fe6No8SCw-aVnCfRi
#Doc from NERSC
#https://docs.nersc.gov/development/languages/python/parallel-python/




def task(rank):
	return 9**(rank+3)

data_set=[]

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

print("size:"+str(size))
print("rank:"+str(rank))


for i in range(size): 
	if 1!=0:
		print("Doing the task of rank"+str(i))
		data=task(rank)		#doing the task
		print("sending from "+str(i)+" to 0")
		print("data:"+str(data))
		data_set.append(data)
		print("data_set"+str(data_set))
		#comm.send(data,dest=0,tag=i)			#send data
		print("---------------------")
	#else:
		#print("************")
		#for j in range(size):
			#print("receive from "+str(j))
			#data=comm.recv(source=j,tag=j)  	#recieve data
			#data_set.append(data)
			#print(str(data_set))
		
		


