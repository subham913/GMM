import numpy as np
import scipy.io
import math
import PIL

mat = scipy.io.loadmat('mnist_small.mat')
tick=1
X=mat['X']
Y=mat['Y']
N,D=X.shape
# KList=[20]
KList=[5,10,15,20]
# print(N)
numIter=200
batches=100
# print(X.shape)
# print(Y.shape)
Classes=[]
for i in range(0,10):

    Z=Y.reshape(Y.shape[0],)
    Z=list(np.where(Z==i))

    temp1=X[Z]
    # print(Z)
    Z[0]=Z[0].reshape(Z[0].shape[0],1)

    temp=np.concatenate((temp1,Z[0]),axis=1)
    # print(temp.shape)
    Classes.append(temp)
# exit(0)


def get_batch(batch_size):
    per_class_sample=int(batch_size/10)
    batch=np.zeros(per_class_sample)
    flag=0

    for C in Classes:

        idx=np.random.randint(C.shape[0],size=per_class_sample)

        if(flag==0):
            batch=C[idx,:]
            flag=1
        else:
            batch=np.concatenate((batch,C[idx,:]),axis=0)
    return batch

def pdf_calc(pik,x,mu,sigma2):
    return((np.log(pik) - np.linalg.norm(mu - x)**2) / (2*sigma2))


def stepwise_EM(K):
    pi=np.ones((K,1))/K

    mu=np.random.randn(K,D)
    z=np.zeros((N,K))
    sigma2=1

    for iter in range(1,10000):
        lr=math.pow((1+tick),-0.55)
        batch=get_batch(batches)
        idx=[]
        X_batch=[]
        for sample in batch:
            n=sample[-1]
            idx.append(n)
            for k in range(0,K):
                # print(pi[k])

                z[n,k]=(pdf_calc(pi[k],sample[:-1],mu[k],sigma2))
            maxZ=np.max(z[n])

            z[n]=np.exp(z[n]-maxZ-math.log(np.sum(np.exp(z[n]-maxZ))))
            X_batch.append(sample[:-1])


        X_batch=np.asarray(X_batch)
        pi_hat=np.sum(z[idx],axis=0)/batches
        pi_hat=np.reshape(pi_hat.shape[0],1)
        # print(pi_hat.shape)

        pi=(1-lr)*pi+lr*pi_hat
        N_list=np.sum(z[idx],axis=0).reshape(-1,1)
        mu_hat=np.dot(z[idx].T,X_batch)
        for k in range(0,N_list.shape[0]):
            if(N_list[k]!=0):
                mu[k]=(1-lr)*mu[k]+(lr*mu_hat[k])/N_list[k]




        tempo=np.linalg.norm(X_batch-np.dot(z[idx],mu))**2

        sigma2_hat=np.linalg.norm(X_batch-np.dot(z[idx],mu))**2/(batches*D)

        sigma2=(1-lr)*sigma2+lr*sigma2_hat

    return mu

def GMM(K):
    pi=np.ones((K,1))/K

    mu=np.random.randn(K,D)
    z=np.zeros((N,K))
    sigma2=1
    # ILL=[]
    for iter in range(1,numIter+1):

        for n in range(0,N):
            for k in range(0,K):

                z[n,k]=(pdf_calc(pi[k],X[n],mu[k],sigma2))
            maxZ=np.max(z[n])
            z[n]=np.exp(z[n]-maxZ-math.log(np.sum(np.exp(z[n]-maxZ))))

        pi=np.sum(z,axis=0)/N

        mu=np.dot(z.T,X)/(np.sum(z,axis=0).reshape(-1,1))
        sigma2=np.linalg.norm(X-np.dot(z,mu))**2/(N*D)

    return mu


for K in KList:
    print("Started for FullBatch EM K= "+str(K))
    mu=GMM(K)
    ctr=0
    for m in mu:
        ctr=ctr+1
        img_j = m.reshape((-1, 28))
        a2 = PIL.Image.fromarray(img_j.astype(np.uint8))
        # a2=a2.rotate(270)
        a2.save('FullBatch_K_'+str(K)+"mu_"+str(ctr)+'.png')

for K in KList:
    print("Started Online EM for K= "+str(K))
    mu=stepwise_EM(K)
    ctr=0
    for m in mu:
        ctr=ctr+1
        img_j = m.reshape((-1, 28))
        a2 = PIL.Image.fromarray(img_j.astype(np.uint8))
        a2.save('online_K_'+str(K)+"mu_"+str(ctr)+'.png')
