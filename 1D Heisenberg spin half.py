import numpy as np
import pylab as plt
import time

tic=time.time()
l=4
f=1
s=[0]*(2**l)

for i in range(2**l):
    s[i]=format(i,"0"+str(l)+"b")

H=np.zeros([2**l,2**l])

for i in range(2**l):
    k=np.zeros(2**l)
    for m in range(l):
        a=np.zeros(l,int)
        a2=''
        if s[i][m]=='0' or s[i][np.mod(m+1,l)]=='1':
            if s[i][m]=='0' and s[i][np.mod(m+1,l)]=='1':
                for p in range(l):
                    if p!=m and p!=np.mod(m+1,l):
                        a[p]=int(s[i][p])
                a[m]=1
                a[np.mod(m+1,l)]=0
                for p in range(l):
                    a2+=str(a[p])
                k[int(a2,2)]+=0.5
        else:
            for p in range(l):
                if p!=m and p!=np.mod(m+1,l):
                    a[p]=int(s[i][p])
            a[m]=0
            a[np.mod(m+1,l)]=1
            for p in range(l):
                a2+=str(a[p])
            k[int(a2,2)]+=0.5
        k[i]+=(int(s[i][m])-0.5)*(int(s[i][np.mod(m+1,l)])-0.5)
    H[:,i]=k[:]
toc=time.time()-tic
print(toc)
tic=time.time()
H=(-1)**f*H

E,V=np.linalg.eigh(H)


toc2=time.time()-tic
print(toc2)


def Sx(i,j,phi1):
    m=np.zeros(2**l)
    for ii in range(2**l):
        if abs(phi1[ii])>10**-6:
            a=np.zeros(l,int)
            a2=''
            if s[ii][i]=='0' and s[ii][j]=='0':
                for p in range(l):
                    if p!=i and p!=j:
                        a[p]=int(s[ii][p])
                a[i]=1
                a[j]=1
                for p in range(l):
                    a2+=str(a[p])
                m[int(a2,2)]+=0.25*phi1[ii]
            if s[ii][i]=='1' and s[ii][j]=='1':
                for p in range(l):
                    if p!=i and p!=j:
                        a[p]=int(s[ii][p])
                a[i]=0
                a[j]=0
                for p in range(l):
                    a2+=str(a[p])
                m[int(a2,2)]+=0.25*phi1[ii]
            if s[ii][i]=='0' and s[ii][j]=='1':
                for p in range(l):
                    if p!=i and p!=j:
                        a[p]=int(s[ii][p])
                a[i]=1
                a[j]=0
                for p in range(l):
                    a2+=str(a[p])
                m[int(a2,2)]+=0.25*phi1[ii]
            if s[ii][i]=='1' and s[ii][j]=='0':
                for p in range(l):
                    if p!=i and p!=j:
                        a[p]=int(s[ii][p])
                a[i]=0
                a[j]=1
                for p in range(l):
                    a2+=str(a[p])
                m[int(a2,2)]+=0.25*phi1[ii]

    return np.dot(phi1,m)

def Sy(i,j,phi1):

    m=np.zeros(2**l)
    for ii in range(2**l):
        if abs(phi1[ii])>10**-6:
            a=np.zeros(l,int)
            a2=''
            if s[ii][i]=='0' and s[ii][j]=='0':
                for p in range(l):
                    if p!=i and p!=j:
                        a[p]=int(s[ii][p])
                a[i]=1
                a[j]=1
                for p in range(l):
                    a2+=str(a[p])
                m[int(a2,2)]+=-0.25*phi1[ii]
            if s[ii][i]=='1' and s[ii][j]=='1':
                for p in range(l):
                    if p!=i and p!=j:
                        a[p]=int(s[ii][p])
                a[i]=0
                a[j]=0
                for p in range(l):
                    a2+=str(a[p])
                m[int(a2,2)]+=-0.25*phi1[ii]
            if s[ii][i]=='0' and s[ii][j]=='1':
                for p in range(l):
                    if p!=i and p!=j:
                        a[p]=int(s[ii][p])
                a[i]=1
                a[j]=0
                for p in range(l):
                    a2+=str(a[p])
                m[int(a2,2)]+=0.25*phi1[ii]
            if s[ii][i]=='1' and s[ii][j]=='0':
                for p in range(l):
                    if p!=i and p!=j:
                        a[p]=int(s[ii][p])
                a[i]=0
                a[j]=1
                for p in range(l):
                    a2+=str(a[p])
                m[int(a2,2)]+=0.25*phi1[ii]

    return np.dot(phi1,m)

def Sz(i,j,phi1):
    m=np.zeros(2**l)
    for ii in range(2**l):
        if abs(phi1[ii])>10**-6:
            m[ii]+=(int(s[ii][i])-0.5)*(int(s[ii][j])-0.5)*phi1[ii]
    return np.dot(phi1,m)

k=[]
for i in range(len(E)):
    if abs(E[i]-min(E))<=10**-8:
        k.append(i)
o=len(k)

phi=np.zeros([o,2**l])
for i in range(o):
    phi[i,:]=V[:,k[i]]




d=np.arange(1,l)
avsx=np.zeros([o,l-1])
avsy=np.zeros([o,l-1])
avsz=np.zeros([o,l-1])
count=np.zeros(l-1)
for jj in range(o):
    for i in range(l-1):
        for j in range(i+1,l):
            avsx[jj,j-i-1]+=Sx(i,j,phi[jj,:])
            avsy[jj,j-i-1]+=Sy(i,j,phi[jj,:])
            avsz[jj,j-i-1]+=Sz(i,j,phi[jj,:])
            if jj==0:
                count[j-i-1]+=1

avsxt=np.zeros([l-1])
avsyt=np.zeros([l-1])
avszt=np.zeros([l-1])

avsxt[:]=np.average(avsx,axis=0)/count[:]
avsyt[:]=np.average(avsy,axis=0)/count[:]
avszt[:]=np.average(avsx,axis=0)/count[:]

avs=avsxt+avsyt+avszt

f=2
if f==0:

    file1=open(str(l)+"E.txt","w")
    for i in range(2**l):
        file1.write('%.5f\n'%E[i])
    file1.close()
    
    # file2=open(str(l)+"V.txt","w")
    # for i in range(2**l):
    #     for j in range(2**l):
    #         file2.write("%.5f\n"%V[i,j])
    # file2.close()
    
    np.savetxt(str(l)+"avsx.txt",avsx,delimiter=',')
    np.savetxt(str(l)+"avsy.txt",avsy,delimiter=',')
    np.savetxt(str(l)+"avsz.txt",avsz,delimiter=',')

## Ferromagnetic
elif f==1:
    file1=open(str(l)+"Ef.txt","w")
    for i in range(2**l):
        file1.write('%.5f\n'%E[i])
    file1.close()
    
    # file2=open(str(l)+"Vf.txt","w")
    # for i in range(2**l):
    #     for j in range(2**l):
    #         file2.write("%.5f\n"%V[i,j])
    # file2.close()
    
    np.savetxt(str(l)+"avsxf.txt",avsxt,delimiter=',')
    np.savetxt(str(l)+"avsyf.txt",avsyt,delimiter=',')
    np.savetxt(str(l)+"avszf.txt",avszt,delimiter=',')


