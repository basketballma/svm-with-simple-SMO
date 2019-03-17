import random
import numpy as np
def load_dataset(filename):
    f=open(filename)
    datamat=[]
    label=[]
    for line in f.readlines():
        new_line=line.strip().split()
        datamat.append([float(new_line[0]),float(new_line[1])])
        label.append(float(new_line[2]))
    return datamat,label

def select_j(i,m):
    j=-1
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def xiuzheng(L,H,aim):
    if aim>H:
        aim=H
    elif aim<L:
        aim=L
    return aim


def smo_simplify(datamat,label,C,toler,max_iter):
    datamat=np.mat(datamat)
    labelmat=np.mat(label).transpose()
    m,n=datamat.shape
    alphas=np.mat(np.zeros((m,1)))
    b=0
    iter=0
    while(iter<max_iter):
        alphasized_changed=0
        for i in range(m):
            
            fxi=float(np.multiply(alphas,labelmat).transpose()*(datamat*datamat[i,:].transpose()))+b
            Ei=fxi-float(labelmat[i])
            if ((labelmat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelmat[i]*Ei>toler)\
               and (alphas[i]>0)):
                j=select_j(i,m)
                fxj=float(np.multiply(alphas,labelmat).transpose()*(datamat*datamat[j,:].transpose()))+b
                Ej=fxj-float(labelmat[j])
                if labelmat[i]!=labelmat[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[i]+alphas[j]-C)
                    H=min(C,alphas[i]+alphas[j])
                if H==L:continue
                alphasIold=alphas[i].copy()
                alphasJold=alphas[j].copy()
                eta=datamat[i]*datamat[i].T+datamat[j]*datamat[j].T-2*datamat[i]*\
                     datamat[j].T
                if eta<=0:continue
                alphas[j]+=labelmat[j]*(Ei-Ej)/eta
                alphas[j]=xiuzheng(L,H,alphas[j])
                if abs(alphas[j]-alphasJold)<0.00001:
                    continue
                alphas[i]+=labelmat[i]*labelmat[j]*(alphasJold-alphas[j])
                b1=b-Ei-labelmat[i]*(alphas[i]-alphasIold)*datamat[i]*datamat[i].T\
                    -labelmat[j]*(alphas[j]-alphasJold)*datamat[i]*datamat[j].T
                b2=b-Ej-labelmat[i]*(alphas[i]-alphasIold)*datamat[i]*datamat[j].T\
                    -labelmat[j]*(alphas[j]-alphasJold)*datamat[j]*datamat[j].T
                if 0<alphas[i]<C:
                    b=b1
                elif 0<alphas[j]<C:
                    b=b2
                else:
                    b=(b1+b2)/2
                alphasized_changed+=1
            
        if alphasized_changed==0:
            iter+=1
        else:
            iter=0
    return alphas,b
                
                
                
                
