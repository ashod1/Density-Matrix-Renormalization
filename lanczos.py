import numpy as np
import pylab as plt
import time
import random
from scipy.sparse.linalg import eigs,eigsh
from scipy.sparse.linalg import LinearOperator

# def Sx(phi1,i1):
#     k=np.zeros([2**(l-1-i1),2**(i1+1)])
#     k[:,:]=np.reshape(phi1,[2**(l-1-i1),2**(i1+1)],'F')[:,:]
#     for i in np.arange(0,2**(i1+1),2):    
#         k[:,[i,i+1]]=k[:,[i+1,i]]
#     k=(1)*np.reshape(k,2**l,'F')
#     return  k

def Sx(phi1,i1):
    k=phi1.reshape(2**(i1),2,2**(l-1-i1))
    k=k[:,[1,0],:]
    return k.reshape(-1)

def Sy(phi1,i1):
    # k=np.zeros(2**l,complex)
    # k[:]=complex(0,Sxnew(Sznew(phi1,i1),i1)-Sznew(Sxnew(phi1,i1),i1))
    return Sx(Sz(phi1,i1),i1)*1j

def Sz(phi1,i1):
    return -2*(s[:,i1]-0.5)*phi1[:]

def H(phi1):
    k=np.zeros(2**l,complex)
    for i in range(l):
        k[:]+=(-1)**f*(Sx(Sx(phi1,np.mod(i+1,l)),i)[:]+Sy(Sy(phi1,np.mod(i+1,l)),i)[:]+Sz(Sz(phi1,np.mod(i+1,l)),i)[:])
    return k




ll=np.arange(4,5)
t=[]
f=1
for l in ll:
    tic=time.time()
    
    s=np.zeros([2**l,l],int)
    for i in range(2**l):
        for j in range(l):
            s[i,j]=format(i,"0"+str(l)+"b")[j]
    
    m=4
    E,V=eigsh(LinearOperator((2**l,2**l),matvec=H),k=m,which='SA')


# G=np.zeros([5,5])
# for i in range(5):
#     for j in range(5):
#         G[i,j]=np.dot(V1[:,i],V[:,j])

# c=np.zeros(2**l)
# c=V1[:,0]
# for i in range(5):
#     c[:]-=np.dot(V1[:,0],V[:,i])*V[:,i]