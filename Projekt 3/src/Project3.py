import xlrd
from scipy.stats import zscore
from methods import *
from pylab import *
from writeapriorifile import *


#Converts Present and Absent into numbers.
def convert(s):
    if s == "Present":
        return 1
    else:
        return 0

#Load dataset
doc = xlrd.open_workbook('../../dataset_sorted.xls').sheet_by_index(0)

size = 463
noAttributes = 9

#Get attributes and classnames
attributeNames = doc.row_values(0,1,noAttributes+1)
attributeNamesCHD = doc.row_values(0,1,noAttributes+1+1)

classLabels = doc.col_values(noAttributes+1,1,size)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))


y = np.mat([classDict[value] for value in classLabels]).T

X = np.mat(np.empty((size-1,noAttributes)))
XCHD =np.mat(np.empty((size-1,noAttributes+1)))

for i, col_id in enumerate(range(1,noAttributes+1+1)):
    if(i < len(attributeNames) and attributeNames[i] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,size)]
        if i < noAttributes:
            X[:,i] = np.mat(temp12).T
        XCHD[:,i] = np.mat(temp12).T
    else:
        if i < noAttributes:
            X[:,i] = np.mat(doc.col_values(col_id,1,size)).T
        XCHD[:,i] = np.mat(doc.col_values(col_id,1,size)).T


M = len(attributeNames) 
N = len(y)
C = len(classNames)


XStandardized = zscore(X, ddof=1)


XPC = getPrincipalComponents(XStandardized)

X2PC = np.copy(XPC)

#XPC = XPC[:,0:2]
#X = X[:,0:2]
gmm(XPC,y,9,C,K=3)

#CVK(X,range(1,51),"diag",3)

#hierarchicalClustering(XPC,y,9,C)

#XBin = convertToBinary(X)
#for i in range(0,len(y)):
#XBin = np.insert(XBin, 0, y[i], axis=1)
#XBin = [XBin[i] + y[i] for i in range(len(XBin))]#s + y#[x + y for x in XBin]

#WriteAprioriFile(XBin,titles=attributeNames,filename="AprioriFile.txt")

#doApriori("AprioriFile.txt",minSup=30,minConf = 55)

#outlierDetection(XStandardized,objects = 5)