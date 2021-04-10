### LIBRARIES

import gudhi
import gudhi.wasserstein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### QUESTION 2: compute landscapes

#piecewise linear funtions
def f_piecewiselinear(b,d,x):
    if x > b and x <=(b+d)/2:
        return x-b
    elif x > (b+d)/2 and x < d:
        return d-x
    else:
        return 0
    
 #function which computes perstitence landscapes from a persistence diagram 
def computePersistentLandscapes(dgm, xmin, xmax, nbnodes, nbld, k):
    n = len(dgm)
    function_values = [[] for i in range(nbnodes)]
    grid = np.linspace(xmin,xmax,nbnodes)
    ls = np.zeros([nbld, nbnodes])
    
    #get the values of piecewise linear functions on the nbodes points of the grid
    for i in range(n):
        (dim,(b,d)) = dgm[i]
        if d == np.inf or k != dim:
            continue
        for j in range(nbnodes):
            function_values[j].append(f_piecewiselinear(b,d,grid[j]))
            
    #sort those values    
    for i in range(nbnodes):
        function_values[i].sort()
    
    #compute landscape
    for i in range(1, nbld + 1):
        ls[i-1]=[l[-i] for l in function_values]
    return ls


### QUESTION 3: reproduce experiments

#preprocessing: function to compute logReturns the csv files
def getLogReturns(csvFile):
    df = pd.read_csv("Data/"+csvFile)
    returns = list(df["Adj Close"])
    returns.reverse()
    logreturns = np.log(returns)[1:] - np.log(returns)[:-1]
    return logreturns

#function to get the dates where we have values in the csv files
def getDates(csvFile):
    df = pd.read_csv(csvFile)
    date = list(df["Date"])
    date.reverse()
    return date[1:]

#function to compute the point clouds from the 4 log-returns time-series
def computePointClouds(w,dowjones_lr,nasdaq_lr,russel_lr,sp500_lr):
    n=len(dowjones_lr)
    pointClouds = [[] for i in range (n-w)]
    for i in range(n-w):
        for j in range(w):
            pointClouds[i].append([dowjones_lr[i+j],nasdaq_lr[i+j],russel_lr[i+j],sp500_lr[i+j]])
    return pointClouds

#function to compute persitance diagrams from the point clouds (using GUDHI)
def computePersistenceDiagrams(pointClouds):
    dgms=[]
    for i in range(len(pointClouds)):
        rips_complex = gudhi.RipsComplex(points=pointClouds[i])
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        dgm = simplex_tree.persistence()
        dgms.append(dgm)
    return dgms

#function to plot the norms of persistence landscapes close to financial meltdowns
def plotLandscapesNorms(landscapes,dates,w):
    nbld=1 #we take the first landscape only
    #L1 norm
    ls_l1 = [np.linalg.norm(ls[nbld-1],ord=1) for ls in landscapes]
    #L2 norm
    ls_l2 = [np.linalg.norm(ls[nbld-1],ord=2) for ls in landscapes]
    #plots
    plt.plot(ls_l1[2000:3100])
    plt.title("L1 norm of landscapes before the 2000 financial crisis (w="+str(w)+")"+"\n \n"+str(dates[2000])+" to "+str(dates[3100]))
    plt.show()
    plt.plot(ls_l2[2000:3100])
    plt.title("L2 norm of landscapes before the 2000 financial crisis (w="+str(w)+")"+"\n \n"+str(dates[2000])+" to "+str(dates[3100]))
    plt.show()
    plt.plot(ls_l1[4245:5176])
    plt.title("L1 norm of landscapes before the 2008 financial crisis (w="+str(w)+")"+"\n \n"+str(dates[4245])+" to "+str(dates[5176]))
    plt.show()
    plt.plot(ls_l2[4245:5176])
    plt.title("L2 norm of landscapes before the 2008 financial crisis (w="+str(w)+")"+"\n \n"+str(dates[4245])+" to "+str(dates[5176]))
    plt.show()
    
#main function, running with the experiment with the window size as only parameter
def runExperiment(w):
    
    #preprocessing
    dowjones_lr=getLogReturns("DowJones.csv")
    nasdaq_lr=getLogReturns("Nasdaq.csv")
    russel_lr=getLogReturns("Russell2000.csv")
    sp500_lr=getLogReturns("SP500.csv")
    dates = getDates("DowJones.csv")[w-1:]
    
    #point clouds
    pointClouds=computePointClouds(w,dowjones_lr,nasdaq_lr,russel_lr,sp500_lr)
    
    #diagrams
    dgms = computePersistenceDiagrams(pointClouds)
    
    #landscapes
    k = 1
    nbnodes = 100
    nbld = 1
    
    xmin = np.inf
    xmax = -np.inf
    for dgm in dgms:
        for point in dgm:
            if point[0] == k:
                if point[1][0] < xmin:
                    xmin = point[1][0]
                if point[1][1] > xmax:
                    xmax = point[1][1]

    landscapes = [computePersistentLandscapes(dgm, xmin, xmax, nbnodes, nbld, k) for dgm in dgms]
    
    #plots
    plotLandscapesNorms(landscapes,dates,w)
    
#### QUESTION 4: other methods

def convert_dgm(dgm, dim):
    new_dgm = []
    for point in dgm:
        if point[0] == dim:
            new_dgm.append([point[1][0], point[1][1]])
    return new_dgm

def smooth_curve(a, size):
    new_a = []
    for i in range(len(a)):
        left = max(0, i-size)
        right = min(len(a), i+size)
        new_a.append(sum(a[left:right])/(right-left))
    return(new_a)
    
def runExperimentWasserstein(w):
    #preprocessing
    dowjones_lr=getLogReturns("DowJones.csv")
    nasdaq_lr=getLogReturns("Nasdaq.csv")
    russel_lr=getLogReturns("Russell2000.csv")
    sp500_lr=getLogReturns("SP500.csv")
    dates = getDates("DowJones.csv")[w-1:]
    
    #point clouds
    pointClouds=computePointClouds(w,dowjones_lr,nasdaq_lr,russel_lr,sp500_lr)
    
    #diagrams
    dgms = computePersistenceDiagrams(pointClouds)
    
    #convert the gudhi format into an array
    converted_dgms = [convert_dgm(dgm, 1) for dgm in dgms]
    
    #list of consecutive distances
    consecutive_bot = [gudhi.bottleneck_distance(converted_dgms[i], converted_dgms[i+1]) for i in range(len(converted_dgms)-1)]
    consecutive_bot_smoothed = smooth_curve(consecutive_bot, 10)
    
    consecutive_wass = [gudhi.wasserstein.wasserstein_distance(np.array(converted_dgms[i]), np.array(converted_dgms[i+1])) for i in range(len(converted_dgms)-1)]
    consecutive_wass_smoothed = smooth_curve(consecutive_wass, 10)
    
    #plot
    plt.plot(consecutive_bot[2000:3100])
    plt.title("Consecutive bottleneck distance between persistence diagram before the 2000 financial crisis (w="+str(w)+")"+"\n \n"+str(dates[2000])+" to "+str(dates[3100]))
    plt.show()
    
    plt.plot(consecutive_bot_smoothed[2000:3100])
    plt.title("Consecutive bottleneck distance (smoothed) between persistence diagram before the 2000 financial crisis (w="+str(w)+")"+"\n \n"+str(dates[2000])+" to "+str(dates[3100]))
    plt.show()
    
    plt.plot(consecutive_wass[2000:3100])
    plt.title("Consecutive Wasserstein distance between persistence diagram before the 2000 financial crisis (w="+str(w)+")"+"\n \n"+str(dates[2000])+" to "+str(dates[3100]))
    plt.show()
    
    plt.plot(consecutive_wass_smoothed[2000:3100])
    plt.title("Consecutive Wasserstein distance distance (smoothed) between persistence diagram before the 2000 financial crisis (w="+str(w)+")"+"\n \n"+str(dates[2000])+" to "+str(dates[3100]))
    plt.show()
