import numpy as np
import pylab as plt
import time
import random
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
np.set_printoptions(precision=5)
from scipy.optimize import curve_fit

f=1
g=1

class MPS:
    def __init__(self,size,system_d,chi):
        
        '''
        -size is just lenght of chain   
        -system_d is the single site Hilbert space dimension
        -chi is bond dimension limit
        
        '''
        
        self.size=size
        self.system_d=system_d
        self.chi=chi
        
    def generate_all_up_or_down(self,direction):
        A=[0]*self.size
        self.A=[0]*self.size
        for i in range(self.size):
            A[i]=np.zeros([1,self.system_d,1])
            if direction=="up":
                A[i][0,0,0]=1
            elif direction=="down":
                A[i][0,1,0]=1
        self.A[:]=A[:]
        self.sh=[0]*self.size
        
    def evaluate_operator(self,MPO1):
        L=self.size
        for i in range(L):
            if i==0:
                l1=np.tensordot(self.A[i],np.tensordot(MPO1.O[i],self.A[i].conj(),axes=([1],[1])),axes=([1],[0]))
            else:
                l1=np.tensordot(l1,np.tensordot(self.A[i],np.tensordot(MPO1.O[i],self.A[i].conj(),axes=([1],[1])),axes=([1],[0])),axes=([1,3],[0,2]))
                l1=np.transpose(l1,(0,2,1,3))
        return l1
    
    def generate_small(self,cnew):
        
        '''
        Does successive SVDS to write the wavefunction in right canonical form
        
        Returns
        -----------
        
        -list A which contains the matrices. for right canonical form, A[0] is the first matrix.
        -List sh which contains the diagonal matrices s.
        '''
        
        l1=self.size
        d=self.system_d
        sh=[0]*l1
        n=d**(l1)
        m=1
        A=[0]*l1
        self.A_initial=[0]*l1
        self.A=[0]*l1
        for i in range(l1):
            phi=np.zeros([int(n/d),m*d],complex)
            phi[:,:]=np.reshape(cnew,[int(n/d),m*d])[:,:]
            
            # phi=np.zeros([2**(i+1),2**(l-(i+1))],complex)
            # phi[:,:]=np.reshape(cnew,[2**(i+1),2**(l-(i+1))],'F')[:,:]
            
            u1,s1,v1=np.linalg.svd(phi,full_matrices=False)
            n=np.size(u1,axis=0)
            m=np.size(u1,axis=1)
            
            sh[l1-1-i]=s1
            
            cnew=np.zeros([n,m],complex)
            cnew[:,:]=np.matmul(u1,np.identity(m)*sh[l1-1-i])[:,:]
            cnew.reshape(-1)
            if i<l1/2:
                b=np.reshape(v1,[d**(i+1),d,d**i])
            else:
                b=np.reshape(v1,[d**(l-i-1),d,d**(l-i)])
            A[l1-1-i]=b     

        self.A[:]=A[:]
        self.sh=sh
        self.A_initial[:]=A[:]
        self.sh_initial=sh
        self.canonical="right"
    
    def check_canonical_right(self):
        test_number=0
        for i in range(self.size):
            test_matrix=np.tensordot(self.A[i],self.A[i].conj(),axes=([1,2],[1,2]))
            if (abs(abs(np.eye(np.size(self.A[i],axis=0)))-abs(test_matrix))>10**-8).all():
                test_number+=1
        if test_number==0:
            print("A is right canonical")
        else:
            print("A is not right canonical")
            print(test_number)
    
    def check_canonical_left(self):
        test_number=0
        for i in range(self.size):
            test_matrix=np.tensordot(self.A[i],self.A[i].conj(),axes=([0,1],[0,1]))
            if (abs(abs(np.eye(np.size(self.A[i],axis=2)))-abs(test_matrix))>10**-8).all():
                test_number+=1
        if test_number==0:
            print("A is left canonical")
        else:
            print("A is not right canonical")
            print(test_number)
         
    def combine_pair(self,l):
        '''
        -l is the position of the first site comprising the pair
        '''
        theta=np.tensordot(self.A[l],self.A[l+1],axes=([2],[0]))
        self.theta=theta
        
    def generate_R(self,MPO):
        '''
        -MPO is the MPO of the hamiltonian
        
        generates a list of R's on all sites
        
        the first 2 indices of the tensor represent the upper left/right indices on the graph.
        indices 3 and 4 represent the MPO left/right indices so that they are always 3x3.
        etc.
        '''
        R=[0]*self.size
        for i in range(self.size):
            R_bottom=np.tensordot(MPO.W[i],self.A[i].conj(),axes=([3],[1]))
            R[i]=np.tensordot(self.A[i],R_bottom,axes=([1],[2]))
        self.R=R
        
    def generate_R_environment(self,l):
        '''

        Parameters
        ----------
        l : position at which to evaluate the environment

        Returns
        -------
        the right environment tensor R saved as an antribute R_environment

        '''
        if l==self.size-1:
            r=self.R[-1]
        else:
            for i in range(l,self.size-1):
                if i==l:
                    r=np.tensordot(self.R[i],self.R[i+1],axes=([1,3,5],[0,2,4]))
                    r=np.transpose(r,(0,3,1,4,2,5))
                else:
                    r=np.tensordot(r,self.R[i+1],axes=([1,3,5],[0,2,4]))
                    r=np.transpose(r,(0,3,1,4,2,5))
        self.R_environment=r
    
    def generate_L_environment(self,l):
        '''

        Parameters
        ----------
        l : position at which to evaluate the environment

        Returns
        -------
        the right environment tensor R saved as an antribute R_environment

        '''
        if l==0:
            L=self.R[0]
        else:
            for i in range(l):
                if i==0:
                    L=np.tensordot(self.R[i],self.R[i+1],axes=([1,3,5],[0,2,4]))
                    L=np.transpose(L,(0,3,1,4,2,5))
                else:
                    L=np.tensordot(L,self.R[i+1],axes=([1,3,5],[0,2,4]))
                    L=np.transpose(L,(0,3,1,4,2,5))
        self.L_environment=L
    
    def generate_and_solve_Heff(self,l,MPO1):
        '''

        Parameters
        ----------
        l : position of the first site comprising the pair.
        
        MPO1: MPO of the Hamiltonian

        Returns
        -------
        Heff saved as an attribute Heff

        '''
        if l==0:
            self.generate_R_environment(l+2)
            Heff=np.tensordot(MPO1.W[l],MPO1.W[l+1],axes=([1],[0]))
            Heff=np.transpose(Heff,(0,3,1,2,4,5))
            Heff=np.tensordot(Heff,self.R_environment,axes=([1],[2]))
            Heff=np.transpose(Heff,(0,1,3,5,2,4,6,7,8,9))       
            self.Heff=Heff
            self.Heff=np.reshape(Heff,[self.system_d**2*self.A[l].shape[0]*self.A[l+2].shape[0],self.system_d**2*self.A[l].shape[0]*self.A[l+2].shape[0]])
            self.e,self.v=eigsh(self.Heff,k=2,which='SA')
            print(self.e)
            self.Q=self.v[:,0].reshape([self.A[l].shape[0],self.system_d,self.system_d,self.A[l+2].shape[0]])
        
        elif (l==self.size-2):
            self.generate_L_environment(l-1)
            Heff=np.tensordot(MPO1.W[l],MPO1.W[l+1],axes=([1],[0]))
            Heff=np.transpose(Heff,(0,3,1,2,4,5))
            Heff=np.tensordot(self.L_environment,Heff,axes=([3],[0]))
            Heff=np.transpose(Heff,(0,1,2,6,8,5,3,4,7,9))   
            self.Heff=Heff
            self.Heff=np.reshape(Heff,[self.system_d**2*self.A[l-1].shape[2]*self.A[l+1].shape[2],self.system_d**2*self.A[l-1].shape[2]*self.A[l+1].shape[2]])
            self.e,self.v=eigsh(self.Heff,k=2,which='SA')
            print(self.e)
            self.Q=self.v[:,0].reshape([self.A[l-1].shape[2],self.system_d,self.system_d,self.A[l+1].shape[2]])
        else:
            self.generate_R_environment(l+2)
            self.generate_L_environment(l-1)
            Heff=np.tensordot(MPO1.W[l],MPO1.W[l+1],axes=([1],[0]))
            Heff=np.transpose(Heff,(0,3,1,2,4,5))
            Heff=np.tensordot(Heff,self.R_environment,axes=([1],[2]))
            Heff=np.tensordot(self.L_environment,Heff,axes=([3],[0]))
            Heff=np.transpose(Heff,(0,1,2,5,7,9,10,11,3,4,6,8,12,13))
            
            
            self.Heff=Heff
            self.Heff=np.reshape(Heff,[self.system_d**2*self.A[l-1].shape[2]*self.A[l+2].shape[0],self.system_d**2*self.A[l-1].shape[2]*self.A[l+2].shape[0]])
            self.e,self.v=eigsh(self.Heff,k=2,which='SA')
            print(self.e)
            self.Q=self.v[:,0].reshape([self.A[l-1].shape[2],self.system_d,self.system_d,self.A[l+2].shape[0]])
    
    def left_svd_Q_and_update_R(self,l,MPO1):
        '''
        

        Parameters
        ----------
        l : position of the first site comprising the pair.
        MPO1 : the hamiltonian in MPO form.

        Returns
        -------
        None.

        '''
        test1=self.Q.reshape(-1)

        test2=np.reshape(test1,[self.Q.shape[0]*self.Q.shape[1],self.Q.shape[2]*self.Q.shape[3]])
        u,s,v=np.linalg.svd(test2,full_matrices=False)
        u=u[:,0:min(self.chi,u.shape[1])]
        self.A[l]=u.reshape([self.Q.shape[0],self.system_d,int(len(u.reshape(-1))/(self.Q.shape[0]*self.system_d))])
        self.sh[l]=s[0:min(self.chi,len((s)))]
        
        R_bottom=np.tensordot(MPO1.W[l],self.A[l].conj(),axes=([3],[1]))
        self.R[l]=np.tensordot(self.A[l],R_bottom,axes=([1],[2]))
        

    def right_svd_Q_and_update_R(self,l,MPO1):
        test1=self.Q.reshape(-1)
        test2=np.reshape(test1,[self.Q.shape[0]*self.Q.shape[1],self.Q.shape[2]*self.Q.shape[3]])
        u,s,v=np.linalg.svd(test2,full_matrices=False)
        v=v[0:min(self.chi,v.shape[0]),:]
        self.A[l+1]=v.reshape([int(len(v.reshape(-1))/(self.system_d*self.Q.shape[3])),self.system_d,self.Q.shape[3]])
        self.sh[l]=s[0:min(self.chi,len((s)))]  
        R_bottom=np.tensordot(MPO1.W[l+1],self.A[l+1].conj(),axes=([3],[1]))
        self.R[l+1]=np.tensordot(self.A[l+1],R_bottom,axes=([1],[2]))
        

class MPO:
    def __init__(self,size,system_d):
        
        '''
        -size is just lenght of chain
        -system_d is the single site Hilbert space dimension
        
        '''
        self.size=size
        self.system_d=system_d
        
    def generate(self,O,positions):
        
        '''
        -O is a list of operators acting on sites
        -positions is a list of site positions. It should correspond with O
         i.e. O[0] acts on the site positions[0]. 0 is first site.
        '''
        MPO=[np.eye(self.system_d)]*self.size
        for i in range(len(positions)):
            MPO[positions[i]]=O[i]
        self.O=MPO
        self.positions=positions
    
    def generate_hamiltonian_transverse_ising(self):
        '''
        Generates a list of MPO's representing the transverse field ising model hamiltonian.
        The indices of each MPO are as [left MPO index, right MPO index, up physical index, down physical index]
        The first element is a row vector and the last a column vector.
        '''
        SzO=np.array([[1,0],[0,-1]])
        SxO=np.array([[0,1],[1,0]])
        Wlist=[]
        d=self.system_d
        
        W1=np.zeros([1,3,d,d])
        W1[0,0]=np.eye(d)
        W1[0,1]=-SzO
        W1[0,2]=-g*SxO
        Wlist.append(W1)
        
        WL=np.zeros([3,1,d,d])
        WL[0,0]=-g*SxO
        WL[1,0]=SzO
        WL[2,0]=np.eye(d)
        
        W=np.zeros([3,3,d,d])
        W[0,:,:,:]=W1[0,:,:,:]
        W[:,-1,:,:]=WL[:,0,:,:]

        for i in range(self.size-2):
            Wlist.append(W)
            
        Wlist.append(WL)
        
        self.W=Wlist

l=64

wf=MPS(l,2,chi=10)
wf.generate_all_up_or_down("down")

SzO=np.eye(2)
SzO[1,1]=-1
SxO=np.zeros([2,2])
SxO[0,1]=1
SxO[1,0]=1
MPO1=MPO(l,2)
MPO1.generate_hamiltonian_transverse_ising()
# MPO1.generate([SzO],[4])

#test=wf.evaluate_operator(MPO1)

wf.generate_R(MPO1)

for j in range(3):
    for i in range(l-1):
        wf.generate_and_solve_Heff(i,MPO1)
        wf.left_svd_Q_and_update_R(i,MPO1)
        
    for i in range(l-1):
        wf.generate_and_solve_Heff(l-2-i,MPO1)
        wf.right_svd_Q_and_update_R(l-2-i,MPO1)
S=np.zeros(l-1)
for i in range(l-1):
    wf.generate_and_solve_Heff(i,MPO1)
    wf.left_svd_Q_and_update_R(i,MPO1)
    S[i]=-sum(wf.sh[i][:]**2*np.log(wf.sh[i][:]**2))
    
def f(x,c,g):
    return (c/6)*np.log((l/np.pi)*np.sin(np.pi*x/l))+2*g

xplot=np.arange(1,l)

param=curve_fit(f,xplot,S)
plt.plot(xplot,S,'*')
plt.plot(xplot,f(xplot,param[0][0],param[0][1]))