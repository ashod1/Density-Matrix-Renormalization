import numpy as np
import pylab as plt
np.set_printoptions(precision=5)

L=50

A=np.zeros([2,2,3])

A[0,1,0]=np.sqrt(2/3)
A[0,0,1]=-np.sqrt(1/3)
A[1,1,1]=np.sqrt(1/3)
A[1,0,2]=-np.sqrt(2/3)

E=np.zeros([4,4])
for i in range(3):
    E+=np.kron(A[:,:,i],A[:,:,i])
    
B=np.kron(A[:,:,0],A[:,:,0])-np.kron(A[:,:,2],A[:,:,2])

# m=[2,4]
# Bn=np.zeros([4,4])
# Bn[:,:]=E[:,:]
# for i in range(L-1):
#     if i==m[0]-1 or i==m[1]-1:
#         Bn=np.matmul(Bn,B)
#     else:
#         Bn=np.matmul(Bn,E)

correlations=np.zeros(20)
for j in range(len(correlations)): 
    m=[2,3+j]
    Bn=np.zeros([4,4])
    Bn[:,:]=E[:,:]
    for i in range(L-1):
        if i==m[0]-1 or i==m[1]-1:
            Bn=np.matmul(Bn,B)
        else:
            Bn=np.matmul(Bn,E)
    correlations[j]=np.trace(Bn)

xplot=np.arange(len(correlations))
plt.plot(xplot,np.log(abs(correlations)),'*')
c=np.polyfit(xplot,np.log(abs(correlations)),1)