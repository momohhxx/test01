import sys,os
os.chdir('H:\\01_self study_flash drive\\AAAschulich\\beyond_schulich\\practice\\scotiabank\\BMO_2022\\risk management and financial institutions\\credit risk modeling an introduction\\code for credit risk model\\')
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
import numpy.linalg as anp

import os 


sys.path.insert(1, 'H:\\01_self study_flash drive\\AAAschulich\\beyond_schulich\\practice\\scotiabank\\BMO_2022\\risk management and financial institutions\\credit risk modeling an introduction\\code for credit risk model\\')

import cmUtilities as util

# below e.g. 1 tries to solve what is value of rho and v (this is actully nu 
# in greek letter, not v) in # formula 4.80 (which is when yn ~ t dist)

# rhoTarget=0.05, which is default correlation;
# below tdTarget is the lambda_D in 2nd formula of 4.80, which 
# the author assigned value 0.02;
# tdValue is the second term of the second formula of 4.80;
# rhoValue is the second term of the first formula of 4.80;
# x[0] is myRho, it is rho of t-dist, this is what we want to find;
# x[1] is v/nu, also what you need to find;
# the tCalibrate() fuction just return two values f1, and f2;
# the smaller f1 and f2 are, the better, because it means rhoValue
# and tdValue is more close to target values; 

# ********************** e.g. 1 *******************************

def tCalibrate(x,myP,rhoTarget,tdTarget):
    if (x[0]<=0) | (x[1]<=0):
        return [100, 100]
    jointDefaultProb = jointDefaultProbabilityT(myP,myP,x[0],x[1])
    rhoValue = np.divide(jointDefaultProb-myP**2,myP*(1-myP))
    tdValue = tTailDependenceCoefficient(x[0],x[1])
    f1 = rhoValue - rhoTarget    
    f2 = tdValue - tdTarget    
    return [f1, f2]

# below nu (i.e. v) in formula 4.80 and rho is rho in 4.80; 
# below tTailDependenceCoefficient() is 2nd formula of 4.80; 
# note both nu and rho is what we want to find; 
def tTailDependenceCoefficient(rho,nu):
    a = -np.sqrt(np.divide((nu+1)*(1-rho),1+rho)) 
    tCoefficient = 2*myT.cdf(a,nu+1)    
    return tCoefficient


# computeP_t is from formula 4.78; 
# computeP_t return a probability which is Pn(G,W), conditional default
# probability;

def computeP_t(p,rho,y,w,nu):
    num = np.sqrt(w/nu)*myT.ppf(p,nu)-np.multiply(np.sqrt(rho),y)
    pZ = norm.cdf(np.divide(num,np.sqrt(1-rho)))
    return pZ

# this below is from formula 10.28 or formula 4.33
def computeP(p,rho,g):
    num = norm.ppf(p)-np.multiply(np.sqrt(rho),g)
    pG = norm.cdf(np.divide(num,np.sqrt(1-rho)))
    return pG


# jointIntegradT is the function to be integrated in formula 4.79; 
# it is Pn(g,w)* Pm(g,w)* f_W(w)* phi(g) 
# note here g and w will be integrated out, so we don't need to input
# g, w in jointDefaultProbabilityT function; 
def jointIntegrandT(g,w,p,q,myRho,nu):
    p1 = computeP_t(p,myRho,g,w,nu)
    p2 = computeP_t(q,myRho,g,w,nu)
    density1 = util.gaussianDensity(g,0,1)
    density2 = util.chi2Density(w,nu)    
    f = p1*p2*density1*density2
    return f


  
# jointDefaultProbabilityT is formula 4.79; 
# it used scipy.integrate function;
# see ref "Scipy-Integrate" for more;

def jointDefaultProbabilityT(p,q,myRho,nu):
    lowerBound = np.maximum(nu-40,2)
    support = [[-10,10],[lowerBound,nu+40]]
    pr,err=nInt.nquad(jointIntegrandT,support,args=(p,q,myRho,nu))
    return pr




if __name__ == "__main__":

# below is t-dist;same result as p190;
  tCalibrate([0.173, 9.6],0.01,0.05,0.02)

# therefore, the t correlation coefficient is 0.173,
# and v parameter is 9.6!;



if __name__ == "__main__":
# below is gaussian dist; (note that v=100 which makes t gaussian dist)
  tCalibrate([0.319, 100],0.01,0.05,0.02)



# monte carlo part for e.g.1 (algorithm 4.11 & 4.12)

# generating t-threshold state variables;

# N means N debtors, M means M trials/simulations;
def getTY(N,M,p,rho,nu):
    # G' is a vector of N rows, each row has same length of M random
    # elements from 0 to 1;
    # but G is M by N;
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    # e is a vector with size M by N
    e = np.random.normal(0,1,[M,N])
    # W is also M by N; W below is (1/W)^0.5 in book, where the 
    # 2nd W in book~ chi_square(nu) distribution, see 4.69 formula;
    W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
    Y = np.multiply(W,math.sqrt(rho)*G + math.sqrt(1-rho)*e)
    # Y is also M by N;
    return Y  


# the t-threshold monte-carlo implementation:

def oneFactorTModel(N,M,p,c,rho,nu,alpha):
    # Y is M by N;
    Y = getTY(N,M,p,rho,nu)
 
    #ppf is inverse of CDF;
    # K is M by 1;
    K = myT.ppf(p,nu)*np.ones((M,1))
    print(Y.shape, K.shape)
    # if Y < K then lossIndicator=True else =False;
    # lossIndicator is also M by N;
    lossIndicator = 1*np.less(Y,K)  
    
    #below seems wrong, so i comment out lossDistribution;
    #lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    
    #add my own code:
    lossdist=[]
    for i in lossIndicator:
      lossdist.append(sum(i)*c)
    lossdist.sort()
    #print(len(lossdist),lossdist)
    
    #el,ul,var,es=util.computeRiskMeasures(M,lossDistribution,alpha)
    el,ul,var,es=util.computeRiskMeasures(M,lossdist,alpha)
    return el,ul,var,es  
  
alpha_value = [0.95,0.97,0.99,0.995,0.999,0.9997,0.9999]


myHome='H:\\01_self study_flash drive\\AAAschulich\\beyond_schulich\\practice\\scotiabank\\BMO_2022\\risk management and financial institutions\\credit risk modeling an introduction\\code for credit risk model\\'
dpFile = myHome+"defaultProbabilties.npy"
expFile = myHome+"exposures.npy"

c = np.load(expFile)  
p = np.load(dpFile)
N = len(c)
print(N)


# but the result is not exactly same with p192, it is comparable though;

# oneFactorTModel(N,M,p,c,rho,nu,alpha);

# oneFactorTModel(100,100000,0.01,10,0.173,9.6,alpha_value)











# e.g. 2 is the multivariate t mixture model in p222;

# ******************************* e.g. 2 ****************************************


default_data = np.load('defaultProbabilties.npy')
region=np.load('regions.npy')

pd.DataFrame(region).shape
pd.DataFrame(region).head(10)


default_data.shape
pd.DataFrame(default_data).head(10)

df01=pd.DataFrame({'region_name':region,'prob':default_data})
df01.head()
df01.groupby('region_name')['prob'].mean()

"""
1.0    0.007997
2.0    0.012210
3.0    0.009581
"""

# below is from appendix foruma A.20 on page 642;

def bivariateTDensity(x1,x2,rho,nu,d=2):
    Sigma = np.array([[1,rho],[rho,1]])
    myX = np.array([x1,x2])
    t1 = math.gamma((nu+d)/2)
    t2 = math.gamma(nu/2) 
    t3 = np.power(nu*math.pi,d/2)
    t4 = np.sqrt(anp.det(Sigma))
    constant = np.divide(t1,t2*t3*t4)
    t5 = np.dot(np.dot(myX,anp.inv(Sigma)),myX)
    integrand = constant*np.power(1+t5/nu,-(nu+d)/2)
    return integrand

# note t_ans is the value of integral, which is a prob of P(X<xx, Y<yy)
# err is error value (a very small number);
def bivariateTCdf(yy,xx,rho,nu):    
    t_ans, err = nInt.dblquad(bivariateTDensity, -10, xx,
                   lambda x: -10,
                   lambda x: yy,args=(rho,nu))
    return t_ans



# below is formula 4.120; R is correlation matrix;
def buildAssetCorrelationMatrix(a,b,regionId):
    # J=3
    J = len(b)
    # below R is initiated as a 3 by 3 matrix with all elements 0;
    R = np.zeros([J,J])
    for n in range(0,J):
        for m in range(0,J):
            if regionId[n]==regionId[m]:
                R[n,m] = a + (1-a)*np.sqrt(b[n]*b[m])
            else:
                R[n,m] = a
    return R



# nu =30, see p222;
# below uses default correlation formula: 

"""
rho(I_Dn,I_Dm)=[ P(D_n & D_m)-Pn*Pm ]/ [Pn*Pm*(1-Pn)*(1-Pm)]^(1/2)
below D is calculated default correlation matrix (3 by 3)
"""

def buildDefaultCorrelationMatrix(a,b,pMean,regionId,nu):
    # J=3
    J = len(regionId)
    R = buildAssetCorrelationMatrix(a,b,regionId)    
    D = np.zeros([J,J])
    for n in range(0,J):
        p_n = pMean[n]
        for m in range(0,J):
            p_m = pMean[m]
            # norm.ppf(0.95)=1.64 for example;
            # below when calculate P_nm we just use t-dist instead of 
            # complex three integral due to both Ym and Yn ~ t-dist;
            # please see my notes on p222 for details about this; 
            p_nm = bivariateTCdf(norm.ppf(p_n),norm.ppf(p_m),R[n,m],nu)
            D[n,m] = (p_nm - p_n*p_m)/math.sqrt(p_n*(1-p_n)*p_m*(1-p_m))
    print("D is: ",D)
    return D


# B is the correlation matrix used in chpt3;
# B should be equal to:

#     A      B     C
#A  0.03   0.02   0.01
#B  0.02   0.04   0.04
#C  0.01   0.04   0.09

# this returns a distance between two matrix: B and D;

def calibrateOF(x,B,pMean,regionId,nu):
    # below a is a in formula 4.116, b is a vector for b1,b2,b3 (3 regions coefficients)
    a = x[0]
    b = np.array([x[1],x[2],x[3]])
    D = buildDefaultCorrelationMatrix(a,b,pMean,regionId,nu)     
    f = anp.norm(D-B,ord='fro')
    return f

# B is the desired target default correlation matrix in chpt3;
def calibrateMFT(B,pMean,regionId,nu):
    myBounds = ((0.001,0.30),(0.001,0.30),(0.001,0.30),
                (0.001,0.30))                            
    M = 100
    # xRandom is 100 by 4 matrix; 
    xRandom = np.random.uniform(0,0.30,[M,4])
    functionValues = np.zeros(M)
    for m in range(0,M):
        functionValues[m] = calibrateOF(xRandom[m,:],B,pMean,regionId,nu)
    newOF=np.min(functionValues)
    # this xStart is nothing but an inital value; you can repalce with 
    # any random 4-digit vector (such as [0.01,0.02,0.03,0.05]); the auther
    # just used an initial value that generates relatively small cost function
    # value. 
    xStart = xRandom[functionValues==newOF]
  
    xhat = scipy.optimize.minimize(calibrateOF, 
                    xStart, args=(B,pMean,regionId,nu), 
                    method='SLSQP', jac=None, bounds=myBounds)   
    return xhat    

#multivariate factor t-dist;
#calibrateMFT(B,pMean,regionId,nu)

nparray=np.array([[0.03,0.02,0.01],[0.02,0.04,0.04],[0.01,0.04,0.09]])
pMean=[0.007997,0.01221,0.009581]
regionId=[1,2,3]

if __name__ == "__main__":
# it takes 20 seconds to run below
  calibrateMFT(nparray,pMean,regionId,30)


# result is: x: array([0.027, 0.042, 0.074, 0.250])
# so a=0.027 (global variable), b1=0.042, b2=0.074, b3=0.25;
# same as table 4.10 in the book;





# *********** monte carlo simulation for multivariate T: book p225 *******;



# this is not 100% satisfactory, but this is usable;  

# below code getMultiFactorY is not correct with rId variable;
# added getMultiFactorY2, with the help of chatgpt; 




def getMultiFactorY2(N,M,p,a,b,rId,nu,isT):
    # G is M by N;
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    # regions is M by 3;
    regions = np.random.normal(0,1,[M,len(np.unique(rId))]) 
    # e is M by N;
    e = np.random.normal(0,1,[M,N])
    #R = regions[:,rId]
    R = regions[np.arange(M)[:, None], rId[None, :]]
    # A is M by N;
    A = np.tile(a*np.ones(N),(M,1))
    B = np.tile(b[rId],(M,1))
    T0 = np.multiply(np.sqrt(A),G)
    T1 = np.sqrt(1-A)
    T2 = np.multiply(np.sqrt(B),R) + np.multiply(np.sqrt(1-B),e)
    if isT==1: 
        W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
        return np.multiply(W,T0+np.multiply(T1,T2))
    else: 
        return T0+np.multiply(T1,T2)





def multiFactorThresholdModel(N,M,a,b,rId,p,c,nu,alpha,isT):
    Y = getMultiFactorY2(N,M,p,a,b,rId,nu,isT)
    if isT==1:
        K = myT.ppf(p,nu)*np.ones((M,1)) 
    else:
        K = norm.ppf(p)*np.ones((M,1))        
    lossIndicator = 1*np.less(Y,K)  
    
    #add my own code:
    lossdist=[]
    for i in lossIndicator:
      lossdist.append(sum(i)*c)
    lossdist.sort()
    #print(len(lossdist),lossdist)
    
    #next line code seems wrong, added my own code;
    #lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    
    el,ul,var,es=util.computeRiskMeasures(M,lossdist,alpha)
    return el,ul,var,es    

region2=[int(i-1) for i in region]

df02=pd.DataFrame({'region_name':region})
df02.head()
df02.groupby('region_name').region_name.count()

region3=(list(range(0,3))*33)+[1]

#notice the more region C obligators, the higher the VaR; which is due to
#region C has higher b coefficient compared with region A and region B;
# e.g. below region4 also has N=100 but with a lot more region C and has
# higher VaR and ES; reason can also be explained in formula 4.120;

#region4=(list(range(0,3))*20)+[2]*40

"""

multiFactorThresholdModel(100,100000,a=0.027,b=np.array([0.042,0.074,0.25]) \
,rId=np.array(region2),p=0.01,c=10,nu=30,alpha=[0.95,0.99,0.9999],isT=1)


multiFactorThresholdModel(100,100000,a=0.027,b=np.array([0.042,0.074,0.25]) \
,rId=np.array(region3),p=0.01,c=10,nu=30,alpha=[0.95,0.99,0.9999],isT=1)

"""

"""
# original code, but seems not correct; 

def getMultiFactorY(N,M,p,a,b,rId,nu,isT):
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    regions = np.random.normal(0,1,[M,len(np.unique(rId))]) 
    e = np.random.normal(0,1,[M,N])
    R = regions[:,rId]
    A = np.tile(a*np.ones(N),(M,1))
    B = np.tile(b[rId],(M,1))
    T0 = np.multiply(np.sqrt(A),G)
    T1 = np.sqrt(1-A)
    T2 = np.multiply(np.sqrt(B),R) + np.multiply(np.sqrt(1-B),e)
    if isT==1: 
        W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
        return np.multiply(W,T0+np.multiply(T1,T2))
    else: 
        return T0+np.multiply(T1,T2)
      
"""














""" 

below code is used for chpt10 purpose (chpt10 needs to import chpt4 to 
use below functions)

"""
# i quote: "getY function is used to generate the state variables Yn."

def getY(N,M,p,rho,nu,isT):
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    e = np.random.normal(0,1,[M,N])
    if isT==1:
        W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
        Y = np.multiply(W,math.sqrt(rho)*G + math.sqrt(1-rho)*e)
    else:
        Y = math.sqrt(rho)*G + math.sqrt(1-rho)*e
    return Y   

# this nvmDensity is from formula 4.99;
# v is the v in 4.99, myA is the a parameter in
# gamma(a,a), whichModel=0;
def nvmDensity(v,x,myA,whichModel):
    t1 = np.divide(1,np.sqrt(2*math.pi*v))
    t2 = np.exp(-np.divide(x**2,2*v))
    if whichModel==0:
        return t1*t2*util.gammaDensity(v,myA,myA)
    elif whichModel==1:
        return t1*t2*util.gigDensity(v,myA)

# mynote: below nvmPpf whichModel is actually from "invCdf = th.nvmPpf(myP,x[1],0)" so whichModel=0;
def nvmPpf(myVal,myA,whichModel):
    r = scipy.optimize.fsolve(nvmTarget,0,args=(myVal,myA,whichModel))
    return r[0]  



def nvmPdf(x,myA,whichModel):
    f,err = nInt.quad(nvmDensity,0,50,args=(x,myA,whichModel)) 
    return f

def nvmCdf(x,myA,whichModel):
    F,err = nInt.quad(nvmPdf,-8,x,args=(myA,whichModel)) 
    return F    

# mynote: x is the default rate you want to evaluate, see formula 4.100;
def nvmTarget(x,myVal,myA,whichModel):
    F,err = nInt.quad(nvmPdf,-8,x,args=(myA,whichModel)) 
    return F-myVal
  

# mynote: this formula is from 4.88;
# in computeP_NVM, p is expected default rate(myP); rho is the rho parameter
# in threshold model (also see formula 4.88 for rho);
# y is the global factor g, v is the v parameter in formula 4.81,
# myA is the a parameter in gamma(a,a);

def computeP_NVM(p,rho,y,v,myA,invCdf):
    num = np.sqrt(1/v)*invCdf-np.multiply(np.sqrt(rho),y)
    pZ = norm.cdf(np.divide(num,np.sqrt(1-rho)))
    return pZ













def getY2r(N,M,p,myRho,rId,nu,P,isT):
    rhoVector = myRho[rId]
    rhoMatrix = np.tile(rhoVector,(M,1))
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    # G is M by N
    e = np.random.normal(0,1,[M,N])
    systematic = np.multiply(np.sqrt(rhoMatrix),G)
    idiosyncratic = np.multiply(np.sqrt(1-rhoMatrix),e)
    if isT==1:
        W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
        Y = np.multiply(W,systematic + idiosyncratic)
    else:
        Y = systematic + idiosyncratic
        # Y is M by N
    return Y    






