import sys,os
sys.path.insert(1, 'H:\\\')
os.chdir('H:\\')
import pandas as pd
pd.set_option('display.max_rows',150)
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import scipy.linalg as la
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy
from scipy.integrate import quad 
from scipy.optimize import fsolve 
from scipy.stats import norm
import math 
import scipy
import scipy.integrate as nInt
from scipy.stats import t as myT
import scipy.linalg as asp
from scipy.optimize import approx_fprime
import chpt4 as th
import markovChain as mc
import time
import mixtureModels as mix
#import thresholdModels as th
import cmUtilities as util
import os 
import assetCorrelation as ac







# ************* simulating a dataset p585 **************** ;

# K is # of credit state (which is 4);

# below M is the P matrix on p583, H is the Q matrix on p583
def cumulateTransitionMatrix(K,M):
    H =  np.zeros([K,K]) 
    for n in range(0,K):
        for m in range(0,K):
            H[m,(K-1)-n] = np.sum(M[m,(K-1)-n:K])
    return H

# H in transformCumulativeTransitionMatrix is the delta matrix on p584;
# M_c in transformCumulativeTransitionMatrix is the output in cumulateTransitioinMatrix();

def transformCumulativeTransitionMatrix(K,M_c):    
    H = np.zeros([K,K])
    for n in range(0,K):
        for m in range(0,K):
            if M_c[n,m]>=0.9999999:  
                H[n,m]=5
            elif M_c[n,m]<=0.0000001:
                H[n,m] = -5
            else:
                H[n,m] = norm.ppf(M_c[n,m])
    return H    

# simulateCorrelatedTransitionData(K=4,N=100,T=30,Pin=matrix_P,
# wStart=weight_vector,myRho=0.2)
def simulateCorrelatedTransitionData(K,N,T,Pin,wStart,myRho): 
    Q = cumulateTransitionMatrix(K,Pin)
    Delta = transformCumulativeTransitionMatrix(K,Q)                    
    Y = np.zeros([N,T]) # latent variables 100*30
    X = np.zeros([N,T]) # credit states  100*30
    allP = np.zeros([N,T]) # default probabilities
    
    # this Xlast is nothing but a vector of size N, with each element either =
    # 1,2,3 as its rating; note we don't consider state 4 as we 
    # assume no defaults initially (i.e. when t=0)
    # this returns a list of N ratings e.g.: [2,3,1,3,3,...,1,2];
    # but why using wStart as cutoff values? (because the book says use these
    # as an example, it doesn't really matter); 
    
    Xlast = mc.initializeCounterparties(N,wStart) # initial states, N*1
    X0 = Xlast
    
    # Plast is the last col of matrix P (matrix P is from formula 10.11);
    # Pin=matrix_P
    Plast = Pin[(Xlast-1).astype(int),-1] 
    for t in range(0,T):
        # missing 2 parameters, added by me; note nu doesn't matter as we 
        # are not using t-dist for simulation;
        # Y[:,t] = th.getY(N,1,Plast,myRho)  
        # below is from formula 4.69
        # Y is 100*30
        Y[:,t] = th.getY(N,1,Plast,myRho,nu=20, isT=0)
        # Y[:, t] is 1 by 100; 
        
        for n in range(0,N):
            # if rating=4 then it never comes back but fixed at rating 4; 
            if Xlast[n] == 4:
                X[n,t] = 4
                continue
            else:
                X[n,t] = migrateRating(Xlast[n],Delta,Y[n,t])
        allP[:,t] = Pin[(Xlast-1).astype(int),-1]
        Plast = allP[:,t]
        Xlast = X[:,t]
    return X,Y,Delta,allP,X0  





def migrateRating(lastX,Delta,myY):
        transitionRow = (lastX-1).astype(int)
        myMap = Delta[transitionRow,:]    
        if myY>=myMap[1]:
            myX = 1
        elif (myY<myMap[1]) & (myY>=myMap[2]):
            myX = 2
        elif (myY<myMap[2]) & (myY>=myMap[3]):
            myX = 3
        elif myY<myMap[3]:
            myX = 4
        return myX

# below matrix P is from formula 10.11;


matrix_P=np.array([[0.96,0.029,0.01,0.001],[0.1,0.775,0.12,0.005],
[0.12,0.22,0.65,0.01],[0,0,0,1]])


weight_vector=np.array([0.4,0.3,0.3,0]);

# X is 100 by 30 matrix with 1,2,or 3;
# Y is also 100 by 30 matrix and it is just the state variable y(n);
# Delta is 4 by 4 matrix on page 584;
# allP is 100 by 30 matrix 





X,Y,Delta,allP,X0=simulateCorrelatedTransitionData(K=4,N=100,T=30,
Pin=matrix_P,wStart=weight_vector,myRho=0.2)

X.shape # 100 * 30
X
Y.shape # 100 * 30
Y
Delta.shape # 4* 4 
Delta
allP.shape # 100 * 30
allP
X0.shape # 100 * 1
X0


# calculate default prob; 

def getSimpleEstimationData(T,X,allP):
    N,T = X.shape
    kVec = np.zeros(T)
    nVec = np.zeros(T)
    pVec = np.zeros(T)
    kVec[0] = np.sum(X[:,0]==4)
    nVec[0] = N
    pVec[0] = np.mean(allP[:,0])
    for t in range(1,T):
        kVec[t] = np.sum(X[:,t]==4)-np.sum(X[:,t-1]==4)
        # if there is a default, then it won't be counted as 
        # denominator;
        nVec[t] = nVec[t-1] - kVec[t-1]
        pVec[t] = np.mean(allP[X[:,t-1]!=4,t])
    return pVec,nVec,kVec  

pVec, nVec, kVec=getSimpleEstimationData(30,X,allP)

print(f'pVec, nVec, kVec:{pVec,nVec,kVec}')

np.mean(pVec) # 0.31%
np.std(pVec) # 0.07%

#sample default rate: result ~ 0.3%;

# on p589 table it shows avg sample default rate is 31 bps!

# below 3 results are similar to p589 table 10.1

np.mean( kVec/nVec) # 0.35%
np.std( kVec/nVec)  # 0.85%
max( kVec/nVec)  # 4%





x=range(1,31)

y=getSimpleEstimationData(30,X,allP)[2]

plt.bar(x,y)
plt.show()
plt.close()

# notice that above chart default is in clusters!



# ************ methods of moments for mixture models *************** ;

def mixtureMethodOfMoment(x,myP,myV,myModel):
    if myModel==0: # Beta-binomial
        M1 = mix.betaMoment(x[0],x[1],1)
        M2 = mix.betaMoment(x[0],x[1],2)
    elif myModel==1: # Logit
        # mynote: args=() specifies the parameters in the func; in this case,
        # x[0] is mu,x[1] is sigma, 1 is momentNumber, last 1 is isLogit;
        M1,err = nInt.quad(mix.logitProbitMoment,-8,8,args=(x[0],x[1],1,1)) 
        M2,err = nInt.quad(mix.logitProbitMoment,-8,8,args=(x[0],x[1],2,1))
    elif myModel==2: # Probit
        M1,err = nInt.quad(mix.logitProbitMoment,-8,8,args=(x[0],x[1],1,0)) 
        M2,err = nInt.quad(mix.logitProbitMoment,-8,8,args=(x[0],x[1],2,0))
    elif myModel==3: # Poisson-gamma
        M1 = mix.poissonGammaMoment(x[0],x[1],1)
        M2 = mix.poissonGammaMoment(x[0],x[1],2)
    elif myModel==4: # Poisson-lognormal
        M1,err = nInt.quad(mix.poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],1,0)) 
        M2,err = nInt.quad(mix.poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],2,0)) 
    elif myModel==5: # Poisson-Weibull
        M1,err = nInt.quad(mix.poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],1,1)) 
        M2,err = nInt.quad(mix.poissonMixtureMoment,0.0001,0.9999,args=(x[0],x[1],2,1)) 
    f1 = M1 - myP 
    # mynote: so M2 is 2nd moment, M1 is first moment(mean); therefore, what this
    # f2 says is { E(x^2)-[E(x)]^2 } should equal to var(x), i.e. myV!
    f2 =(M2-M1**2) - myV
    if abs(f1)<0.001:
        print("M1 and sigma(std dev): ",round(M1,5),round((M2-M1**2)**0.5,5))
    return [1e4*f1, 1e4*f2]

   




# beta binomial mixture model: 
# below myP and myV are from table 10.1: 

myP=0.348/100 
myV=(0.72/100)**2
myModel=0

#a=scipy.optimize.fsolve(mixtureMethodOfMoment,x0=np.array([0.05,0.16]),args=(myP,myV,myModel))
#a
# so default prob=E(Z)=alpha/(aphla+beta)=a[0]/(a[0]+a[1])=0.0035, same as book;
#a[0]/(a[0]+a[1])

# so variance of default rate is alpha*beta/(alpha+beta)^2*(1+alpha+beta)=
# 0.000052, std is 0.0072, same as book;

#( a[0]*a[1]/( (1+a[0]+a[1])*((a[0]+a[1])**2) ))**0.5


# logit and probit model: result same as book but i am not sure how to get
# mean default rate and std dev of default rate for probit model? 
# mixtureMethodOfMoment(x=np.array([-6,2]),myP=0.348/100,myV=0.72/100,myModel=1)

myP=0.348/100
myV=(0.72/100)**2

myModel=1

#b=scipy.optimize.fsolve(mixtureMethodOfMoment,x0=[3,4],args=(myP,myV,myModel))

#b

# poisson gamma model:

myP=0.348/100
myV=(0.72/100)**2

myModel=3

c=scipy.optimize.fsolve(mixtureMethodOfMoment,x0=[3,4],args=(myP,myV,myModel))

c

# mean default prob is about a/b which is 0.0035, so it is ok;
# below formulas are from p120 "a quick and dirty calibration";

c[0]/c[1]

# variance is about a/b^2 which is 0.000054,std is (0.000054)^0.5=0.0073, which is fine;

( c[0]/(c[1]*c[1]) ) **0.5


# ************* methmod of moments for threshold models **************;
"""
in the context of t dist, below thresholdMoment() returns what is similar to
the func inside formula 4.79, ie inside below double integral:
integral_intergral[ Pn(g,w)*Pm(g,w)*f(W)*phi(G)] dwdg
f(W)~chisquare(nu), phi(G)~standard normal dist; but with a little change
that we added myMoment parameter, so when myMoment =1 it is same as 4.79
but when my Moment=2 it should be like 10.24 (10.24 without the -mu(p) term);

in th.computeP_t, myP is the default rate we setup, p1 is rho (ie correlction
coefficient for t-dist), p2 is nu. g, w is the global systematic factor and 
the chi-square dist variable;

in variance-gamma model, p1 (ie x[0])is the correlation in normal-variance mixture models (formula 4.81),
p2 (ie x[1]) is the a parameter in gamma(a,b) dist; note in
variance-gamma, a=b (see sec 4.5.4 for more);
"""

# why invCdf=0 in thresholdMoment for variance-gamma? is it because invCdf is redefined in getThresholdMoments?;
# yes, it is!

def thresholdMoment(g,w,p1,p2,myP,whichModel,myMoment,invCdf=0):
    d1 = util.gaussianDensity(g,0,1)
    if whichModel==1: # t
        d2 = util.chi2Density(w,p2)
        integrand = np.power(th.computeP_t(myP,p1,g,w,p2),myMoment)
    if whichModel==2: # Variance-gamma
        # mynote: here p2 is the 'a' parameter in gamma(a,a) dist, see section 4.5.4 for details about gamma(a,a);
        d2 = util.gammaDensity(w,p2,p2)
        integrand = np.power(th.computeP_NVM(myP,p1,g,w,p2,invCdf),myMoment)
    if whichModel==3: # Generalized hyperbolic
        d2 = util.gigDensity(w,p2)
        integrand = np.power(th.computeP_NVM(myP,p1,g,w,p2,invCdf),myMoment)
    return integrand*d1*d2


"""
below getThresholdMoments(),
for t model, x[0] is p1 (rho in t-dist); x[1] is nu; myP is expected default rate;
for variance-gamma, x[0] is correlation in normal-variance mixture threshold model,
x[1] is a in v~gamma(a,a) see formula 10.23 for more;
"""

def getThresholdMoments(x,myP,whichModel):
    if whichModel==0: # Gaussian
        M1,err = nInt.quad(integrateGaussianMoment,-5,5,args=(x[0],myP,1)) 
        M2,err = nInt.quad(integrateGaussianMoment,-5,5,args=(x[0],myP,2)) 
    elif whichModel==1: # t
        lowerBound = np.maximum(x[1]-40,2)
        support = [[-7,7],[lowerBound,x[1]+40]]
        M1,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,1))
        M2,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,2))
    elif whichModel==2: # Variance-gamma
        invCdf = th.nvmPpf(myP,x[1],0)
        # mynotes: the [-7,7] is for g's support (normal dist);
        # the [0,100] is for v's support (gamma dist >0);
        support = [[-7,7],[0,100]]
        M1,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,1,invCdf))
        M2,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,2,invCdf))
    elif whichModel==3: # Generalized hyperbolic        
        invCdf = th.nvmPpf(myP,x[1],1)
        support = [[-7,7],[0,100]]
        M1,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,1,invCdf))
        M2,err=nInt.nquad(thresholdMoment,support,args=(x[0],x[1],myP,whichModel,2,invCdf))
    return M1,M2


# below myP is expected default rate, myV is expected variance; 
def thresholdMethodOfMoment(x,myP,myV,whichModel):
    print("Evaluating at x =", x)
    if (x[0]<=0) | (x[0]>1):
        return 100
    M1,M2 = getThresholdMoments(x,myP,whichModel)
    f1 = M1 - myP    
    f2 =(M2-M1**2) - myV
    print("M1, Variance is: ",round(M1,5),round(M2-M1**2,5))
    return [1e4*f1,1e4*f2]

x_vector=[0.05, 12]

myP=0.0035 
myV=0.000054 
whichModel=1


#thresholdMethodOfMoment([0.02, 8],0.0035,0.000054,1)

# below opt_moment_t run for t model takes more than 20 min to run! ;
# but result is very close to book (book p1=0.11 p2=26.2; 
# mine p1=0.11 p2=25.8 after ~20 min;)

#opt_moment_t=scipy.optimize.fsolve(thresholdMethodOfMoment,x0=x_vector,args=(myP,myV,whichModel))



def integrateGaussianMoment(g,r,myP,myMoment):
    integrand = np.power(th.computeP(myP,r,g),myMoment)
    return integrand*util.gaussianDensity(g,0,1)


x_vector=[0.03, 3]

myP=0.0035 
myV=0.000054 
whichModel=2


# below opt_moment_variance_gamma takes more than 8 min to run; only run success using IDLE, using R not succeeded;
# result is p1=0.1 p2=7.2;

#opt_moment_variance_gamma=scipy.optimize.fsolve(thresholdMethodOfMoment,x0=x_vector,args=(myP,myV,whichModel))


#thresholdMethodOfMoment([0.8, 20],myP,myV,whichModel)




















#**************************** MLE methods ********************************;

# getCMF (conditional mass function) is from formula 10.30 (inside integral!)
def getCMF(g,myRho,myP,myN,myK):
    pg = th.computeP(myP,myRho,g)
    # mynote: getBC is nothing but combinations (e.g. getBC(6,2)=15);
    # below int function add by me;
    myN=int(myN)
    myK=int(myK)
    f=util.getBC(myN,myK)*np.power(pg,myK)*np.power(1-pg,myN-myK)
    cmf = f*util.gaussianDensity(g,0,1)        
    return cmf

def logLSimple(x,T,pVec,nVec,kVec):
    L = np.zeros(T)
    for t in range(0,T):
        # in quad(f(x),...) the variable needs to be integrated is the first
        # variable in f(x), in this case the first variable in getCMF which
        # is g! the x in quad args=() is the first non-integration variable 
        # in getCMF(), which is myRho! 
        #L[t],err = nInt.quad(getCMF,-5,5,args=(x,pVec[t],nVec[t],kVec[t]))
        L[t],err = nInt.quad(getCMF,-5,5,args=(x,pVec[t],nVec[t],kVec[t]),epsabs=1e-5)
        # gpt says: L[t] can easily return 0 when nVec[t] is large or kVec[t]
        # is small, then logL will be -inf...
        # therefore adding a floor to L[t]
        L[t]=max(L[t],1e-100)
    logL = np.sum(np.log(L))
    return -logL          



T=30

pVec,nVec,kVec = getSimpleEstimationData(T,X,allP)   

# below scipy.optimize.minimize() func is very slow, takes about 5 min to calc; sometimes it failed to run...;
bounds=[(1e-5, 0.999)]
rHat = scipy.optimize.minimize(logLSimple,x0=0.1,args=(T,pVec,nVec,kVec),method='TNC',bounds=bounds)

rHat

# result is x:array([0.285]), also result is x:array([0.115])

# try to get table 10.5 numbers: 

X,Y,Delta,allP,X0=simulateCorrelatedTransitionData(K=4,N=100,T=30,
Pin=matrix_P,wStart=weight_vector,myRho=0.45)


pVec,nVec,kVec = getSimpleEstimationData(T,X,allP)   

bounds=[(1e-5, 0.999)]
rHat = scipy.optimize.minimize(logLSimple,x0=0.1,args=(T,pVec,nVec,kVec),method='TNC',bounds=bounds)

rHat


def computeSimpleScore(x0,T,pVec,nVec,kVec):
    h = 0.00001
    fUp = logLSimple(x0+h/2,T,pVec,nVec,kVec)
    fDown = logLSimple(x0-h/2,T,pVec,nVec,kVec)    
    score = np.divide(fUp-fDown,h)
    return score

round( computeSimpleScore(rHat.x[0],30,pVec,nVec,kVec),4)

# result is 0 (which is expected)

fInf = ac.simpleFisherInformation( computeSimpleScore(rHat.x[0],30,pVec,nVec,kVec))
se = np.sqrt(-np.reciprocal(fInf))





# table 10.6 results:

def getCMF_CRplus(z,a,w,myP,myN,myK):
    
    pg = myP*(1-w+w*z)
    # mynote: getBC is nothing but combinations (e.g. getBC(6,2)=15);
    # below int function add by me;
    myN=int(myN)
    myK=int(myK)
    f=util.getBC(myN,myK)*np.power(pg,myK)*np.power(1-pg,myN-myK)
    cmf =f* util.gammaDensity(z,a,a)
    return cmf


def logLSimple2(x,T,a, pVec,nVec,kVec):
    L = np.zeros(T)
    for t in range(0,T):
        L[t],err = nInt.quad(getCMF_CRplus,0,100,args=(a,x,pVec[t],nVec[t],kVec[t]),epsabs=1e-5)
      
        L[t]=max(L[t],1e-100)
    logL = np.sum(np.log(L))
    return -logL  

T=30
pVec,nVec,kVec = getSimpleEstimationData(T,X,allP)   



def computeSimpleScore2(x0,T,a,pVec,nVec,kVec):
    h = 0.00001
    fUp = logLSimple2(x0+h/2,T,a,pVec,nVec,kVec)
    fDown = logLSimple2(x0-h/2,T,a,pVec,nVec,kVec)    
    score = np.divide(fUp-fDown,h)
    return score

def simpleFisherInformation(x0,T,a,pVec,nVec,kVec):
    h = 0.000001
    f = logLSimple2(x0,T,a,pVec,nVec,kVec)    
    fUp = logLSimple2(x0+h,T,a,pVec,nVec,kVec)
    fDown = logLSimple2(x0-h,T,a,pVec,nVec,kVec)    
    I = -np.divide(fUp-2*f+fDown,h**2)
    return I

bounds=[(1e-5, 0.999)]

a=0.05
wHat = scipy.optimize.minimize(logLSimple2,x0=0.4,args=(T,a,pVec,nVec,kVec),method='TNC',bounds=bounds)
wHat.x[0] # 0.61 (vs 0.44 on book)
wHat.fun  # 18
# a value of 18 means the average probability per year is e^{-18/30} ~ 0.54
# score function, the closer to 0 the better. 
round( computeSimpleScore2(rHat.x[0],30,a, pVec,nVec,kVec),4) # -0.0008
finf = simpleFisherInformation(wHat.x[0],30,a,pVec,nVec,kVec) 
finf # result -474; 

se = np.sqrt(-np.reciprocal(finf))
se # 0.046

a=0.125
wHat = scipy.optimize.minimize(logLSimple2,x0=0.4,args=(T,a,pVec,nVec,kVec),method='TNC',bounds=bounds)
wHat.x[0]  # 0.46 (vs 0.62 on book)
wHat.fun   # 18.1
round( computeSimpleScore2(rHat.x[0],30,a, pVec,nVec,kVec),4) # -1.58
finf = simpleFisherInformation(wHat.x[0],30,a,pVec,nVec,kVec) 
finf # result -97; 

se = np.sqrt(-np.reciprocal(finf))
se # 0.1


a=0.25
wHat = scipy.optimize.minimize(logLSimple2,x0=0.4,args=(T,a,pVec,nVec,kVec),method='TNC',bounds=bounds)
wHat.x[0]  # 0.87 (vs 0.8 on book)
wHat.fun  # 18.4
round( computeSimpleScore2(rHat.x[0],30,a, pVec,nVec,kVec),4) # -2.8
finf = simpleFisherInformation(wHat.x[0],30,a,pVec,nVec,kVec) 
finf # result -35; 

se = np.sqrt(-np.reciprocal(finf))
se # 0.17

# will choose the min log likelihood value because we put a '-' sign 
# in front of logL! also choose score value closer to 0 and lowest 
# standard error for w!

# therefore choose a=0.05












# ************* algorithm 10.8 two-parameter regional log-likelihood functions *************;
# can't get same numbers as the book! (actully it is same result, see comments below 20251228)
# it takes 10 min to run below code, very slow;
# this code can be very useful... 

def initializeRegion2r(N,rStart):
    startRegion = np.zeros(N)
    w = np.cumsum(rStart)
    u = np.random.uniform(0,1,N)
    for n in range(0,N):
        if ((u[n]>0) & (u[n]<=w[0])):
            startRegion[n] = 0
        elif ((u[n]>w[0]) & (u[n]<1)):
            startRegion[n] = 1
    return startRegion  

# initializeRegion2r(10,0.7) 

# result: array([0., 0., 0., 1., 0., 1., 1., 0., 0., 0.])

matrix_P=np.array([[0.96,0.029,0.01,0.001],[0.1,0.775,0.12,0.005],
[0.12,0.22,0.65,0.01],[0,0,0,1]])

weight_vector=np.array([0.4,0.3,0.3,0]);

# createRatingData2r(K=4,N=100,T=30,P=matrix_P,wStart=weight_vector,rStart=r_start_value, myRho=np.array([0.15,0.45]),nu=0.2, isT=0)
# rStart=0.7

def createRatingData2r(K,N,T,P,wStart,rStart,myRho,nu,isT): 
    Q = cumulateTransitionMatrix(K,P)
    Delta = transformCumulativeTransitionMatrix(K,Q)
    rId = initializeRegion2r(N,rStart).astype(int)
    # rId is N*1
    Y = np.zeros([N,T]) # latent variables
    X = np.zeros([N,T]) # credit states
    allP = np.zeros([N,T]) # default probabilities
    Xlast = mc.initializeCounterparties(N,wStart) # initial states
    X0 = Xlast 
    # Plast is N*1
    Plast = P[(Xlast-1).astype(int),-1] 
    for t in range(0,T):
        # Y is the Yn state variable, 1 * N; 
        # same Yn for T=t;
        Y[:,t] = th.getY2r(N,1,Plast,myRho,rId,nu,P,isT)    
        for n in range(0,N):
            if Xlast[n] == 4:
                X[n,t] = 4
                continue # note, you can remove 'continue' here
            else:
                X[n,t] = migrateRating(Xlast[n],Delta,Y[n,t])
        allP[:,t] = P[(Xlast-1).astype(int),-1]
        Plast = allP[:,t]
        Xlast = X[:,t]
    return X,Y,Delta,allP,X0,rId   

# note, nu is only used for T dist;


d=createRatingData2r(K=4,N=20,T=10,P=matrix_P,wStart=weight_vector,rStart=0.7, myRho=np.array([0.15,0.45]),nu=0.2, isT=0)
d[0]
d[1]
d[2]
d[3]
d[4]


'''
chatgpt: in get2rEstimationData P(t)=new defaults/ survivors 
With PD ~ 0.005 and N=100
expected defaults per year ~ 0.5
but simulation often gives even less, especially early in time when rating 
upgrades/downgrades shift exposures.
When the number of survivors shrinks over time (as defaults accumulate),
sometimes no defaults occur at all.
therefore sample default rates <> true default probabilities.
'''


def get2rEstimationData(T,X,X0,rId,allP,numP):
    N,T = X.shape
    kMat = np.zeros([T,numP])
    nMat = np.zeros([T,numP])
    pMat = np.zeros([T,numP])
    for m in range(0,numP):
        xLoc = (rId==m).astype(bool)
        kMat[0,m] = np.sum(X[xLoc,0]==4)
        nMat[0,m] = np.sum(xLoc)
        pMat[0,m] = np.mean(allP[xLoc,0])
        for t in range(1,T):
            kMat[t,m] = np.sum(X[xLoc,t]==4)-np.sum(X[xLoc,t-1]==4)
            nMat[t,m] = nMat[t-1,m] - kMat[t-1,m]
            if np.sum(xLoc)==0:
                pMat[t,m] = 0.0                
            else:
                pMat[t,m] = np.mean(allP[(X[xLoc,t-1]!=4).astype(int),t])
    return pMat,nMat,kMat   # each pMat, nMat, kMat is 30 by 2!





r_start_value=0.7
# note, below 0.15 and 0.45 is from book p610;

T=30

X,Y,Delta,allP,X0,rId=createRatingData2r(K=4,N=100,T=T,P=matrix_P,
wStart=weight_vector,rStart=r_start_value, myRho=np.array([0.15,0.45]),
nu=0.2, isT=0)



pMat,nMat,kMat = get2rEstimationData(T,X,X0,rId,allP,2)


'''
gpt says: pMat[:,0] = MODEL-IMPLIED DEFAULT PROBABILITIES
pMat comes directly from the transition matrix P
It is the probability of default conditional on the rating at time t.
e.g. If a borrower is AAA then PD = 0.001, If AA then PD = 0.005. 
And pMat[t,m] is just the average of those PDs for region m at time t.

(kMat / nMat) = REALIZED DEFAULT FREQUENCY, 
kMat[t] = new defaults during year t
nMat[t] = number of survivors at beginning of t
(kMat / nMat) = actual default RATE that occurred
= empirical default frequency

This is NOT the true PD.
This is just a sample statistic, and can be extremely noisy...
In Real Bank Practice - Which Do We Use?
answer: Use model PD (pMat equivalent)

'''



np.mean(pMat[:,0]) # 0.0051
np.mean(pMat[:,1]) # 0.0051

# do not use below sample mean (too noisy, from chatgpt's comments);

np.mean(  (kMat/nMat)[:,0]) # 0.0036
np.mean(  (kMat/nMat)[:,1]) # 0.00356



# table 10.48 is using below numbers to calculate; 
# but use longer T and bigger N, (N=3000, T=60)
# gpt says: In real data, you NEVER have 'allP', allP exists only 
# in simulation, because the model knows the true transition matrix.
# in real portfolios, you only have k(t), n(t) (counts)
# In real risk management practic, use:
# PD = model PD (e.g., logistic model, credit scorecard, rating system)
# real default counts (sample PDs) are: 
# extremely noisy, biased downward if few defaults
# unstable from year to year, heavily affected by rating migration
# not 'true PDs'
# Your simulator showed exactly this:
# true PD ~ 0.005, sample PD ~ 0.00117


np.mean(allP[:,0])  # 0.0049
np.mean(allP[:,1])  # 0.0090

np.std(allP[:,0]) # 0.0038
np.std(allP[:,1]) # 0.0655

# above is similar with p610 comments;
# note the 2nd default prob has big std dev!;

'''
gemini: when you are working with 2 regions (e.g., North America and Europe), 
the inputs you pass into getProdCMF via the args in quad are no longer 
single numbers; they are vectors of length 2:
  
myP: [p_region1, p_region2]
myN: [n_region1, n_region2]
myK: [k_region1, k_region2]

'''

def getProdCMF(g,myRho,myP,myN,myK):
    pg = th.computeP(myP,myRho,g)
    # Result: pg is a vector: [pg_region1, pg_region2]
    # add by me:
    myK=myK.astype(int)
    myN=myN.astype(int)
    return np.multiply(util.getBC(myN,myK),np.power(pg,myK)*np.power(1-pg,myN-myK))

# cumulative mass function (CDF for discrete variables)
def getCMF2r(g,myRho,pVec3,nVec3,kVec3):
    myF=getProdCMF(g,myRho,pVec3,nVec3,kVec3)
    # np.prod e.g.: np.prod(np.array([10, 2, 5]))=100
    return np.prod(myF)*util.gaussianDensity(g,0,1) # formula 10.45
   
def logL2r(x,T,pMat,nMat,kMat):
    L = np.zeros(T)
    for t in range(0,T):
        # pMat, nMat, and kMat will each have 2 columns (one for each region).
        val,err = nInt.quad(getCMF2r,-5,5,args=(x,pMat[t,:],nMat[t,:],kMat[t,:]))
        if val <= 0 or np.isnan(val):
            L[t] = 1e-12 # A tiny number to keep the log function happy
        else:
            L[t] = val
    return -np.sum(np.log(L))

T=30

bnds = ((1e-4, 0.99), (1e-4, 0.99))

rHat_region= scipy.optimize.minimize(logL2r,x0=np.array([0.1,0.2]),args=(T,pMat,nMat,kMat),method='TNC', bounds=bnds)

print( rHat_region )

# my result isn't close to book, book result is 0.16, 0.27...
# my result: 0.28, 0.09

# my comments: 
# but if you change your N=3000 and T=50 then result is 0.16, 0.35 much closer to true result (0.15, 0.45)!
# but if you change your N=6000 and T=60 then result is 0.16, 0.40 even mroe closer to true result (0.15, 0.45)!


# below code is from gemini;

def computeMultiScore(x0, T, pMat, nMat, kMat):
    h = 0.00001
    n_params = len(x0)
    scores = np.zeros(n_params)
    
    for i in range(n_params):
        # Create a copy of the current parameters
        xUp = x0.copy()
        xDown = x0.copy()
        
        # Wiggle ONLY parameter i
        # note that xUp is a vector. When i=0, only the first 
        # element of that vector changes. When i=1, only the 
        # second element changes.
        xUp[i] += h/2
        xDown[i] -= h/2
        
        # Calculate Likelihoods for this specific wiggle
        fUp = logL2r(xUp, T, pMat, nMat, kMat)
        fDown = logL2r(xDown, T, pMat, nMat, kMat)
        
        # Store the partial derivative
        scores[i] = (fUp - fDown) / h
        
    return scores

# Call it using your optimized results
final_scores = computeMultiScore(rHat_region.x, 30, pMat, nMat, kMat)
print(f"Score for Rho 1: {final_scores[0]}") 

# result: -0.00012

print(f"Score for Rho 2: {final_scores[1]}")

# result: -6.3*e-05

# fisher information: 
# To calculate the Fisher Information Matrix for a two-parameter model, 
# we need to calculate the Hessian 
# (the matrix of second-order partial derivatives) 
# of the log-likelihood function.





# below default correlation is from gemini

from scipy.stats import mvn, norm
import numpy as np

p1, p2 = 0.0049, 0.009 ;
rho_asset = 0.2598  # sqrt(0.15 * 0.45), formula 4.12


# 1. Get thresholds (Z-scores)
z1 = norm.ppf(p1) 
z2 = norm.ppf(p2) 
print(f'z1, z2 {z1, z2}')

# 2. Calculate Joint Probability P(D1 and D2)
# mvn.mvnun integrates the bivariate density
low = np.array([-10, -10]) # approximation for -infinity
up = np.array([z1, z2])
mu = np.array([0, 0])
sigma = np.array([[1, rho_asset], [rho_asset, 1]])

# below from formula 4.23
p_joint, _ = mvn.mvnun(low, up, mu, sigma)

# 3. Calculate Default Correlation
std_d1 = np.sqrt(p1 * (1 - p1))
std_d2 = np.sqrt(p2 * (1 - p2))

# from formula 4.22
rho_default = (p_joint - (p1 * p2)) / (std_d1 * std_d2)

print(f"Joint Probability: {p_joint:.5f}") # 0.00024
print(f"Default Correlation: {rho_default:.4f}") # 0.029, close to 10.48




# below is from author github (same result as above):
# gemeni: The first code uses a Geometric approach (finding the volume 
# under a 2D surface), while the second code uses a Factor Model 
# approach (conditioning on the economy).
# In a Merton model, if we know exactly what the state of the economy G is, 
# the firms become independent. 
# Therefore, the probability that both firms default given G is simply:
# P(D1 and D2 | g) =P1(g) * P2(g) 

# thus, P(D1,D2)=integral(-inf, +inf) P(D1|g)*P(D2|g)* phi(G)* dg


def jointIntegrandRegion(g,p,q,rhoVec):
    p1 = th.computeP(p,rhoVec[0],g)
    p2 = th.computeP(q,rhoVec[1],g)
    density = util.gaussianDensity(g,0,1)
    f = p1*p2*density
    return f
  
def jointDefaultProbabilityRegion(p,q,rhoVec):
    pr,err=nInt.quad(jointIntegrandRegion,-10,10,args=(p,q,rhoVec))
    return pr


def getRegionalDefaultCorrelation(rhoVec,myP):
    jp = jointDefaultProbabilityRegion(myP[0],myP[1],rhoVec)
    return np.divide(jp-myP[0]*myP[1],np.sqrt(myP[0]*(1-myP[0]))*np.sqrt(myP[1]*(1-myP[1])))


getRegionalDefaultCorrelation([0.15,0.45],[0.0049,0.009]) # 0.029
getRegionalDefaultCorrelation([0.15,0.15],[0.0049,0.0049]) # 0.01
getRegionalDefaultCorrelation([0.45,0.45],[0.009,0.009]) # 0.093


# which is very close to 10.48 numbers; 



getRegionalDefaultCorrelation([0.15,0.45],[0.0051,0.0051]) # 0.025
getRegionalDefaultCorrelation([0.15,0.15],[0.0051,0.0051]) # 0.01
getRegionalDefaultCorrelation([0.45,0.45],[0.0051,0.0051]) # 0.075

# above is comparable to 10.49 numbers;












