import numpy as np
import math
import csv
import pdb


def read_data(filename):
	with open(filename,'r') as csvfile:
		datareader = csv.reader(csvfile)
		metadata = next(datareader)
		traindata=[]
		for row in datareader:
			traindata.append(row)

	return (metadata, traindata)

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	testset = list(dataset) #just to convert to list type
	i=0
	while len(trainSet) < trainSize:
		trainSet.append(testset.pop(i)) #pop is used to remove the element used for training
	return [trainSet, testset]

def classify(data,test):
	total_size = data.shape[0]
	print("training data size=",total_size)
	print("test data size=",test.shape[0])
	target=np.unique(data[:,-1]) #contains only no, yes
	count = np.zeros((target.shape[0]),dtype=np.int32)
	prob = np.zeros((target.shape[0]),dtype=np.float32)
	print("target	count	probability")
	for y in range(target.shape[0]):
		for x in range(data.shape[0]):
			if data[x,data.shape[1]-1] == target[y]:
				count[y] +=1
		prob[y]=count[y]/total_size
		print(target[y],"\t",count[y],"\t",prob[y])

	prob0 =np.zeros((test.shape[1]-1),dtype=np.float32)
	prob1 =np.zeros((test.shape[1]-1),dtype=np.float32)
	accuracy=0
	print("instance prediction	target")
	for t in range(test.shape[0]):
		for k in range (test.shape[1]-1): #we don't want to reach the traget value column
			count1=count0=0
			for j in range (data.shape[0]):
				if test[t,k]== data[j,k] and data[j,data.shape[1]-1]==target[0]: #if one of the word in test is found with any word of data increment one for the target yes or no
					count0+=1
				elif test[t,k]==data[j,k] and data[j,data.shape[1]-1]==target[1]:
					count1+=1
			prob0[k]=count0/count[0]
			prob1[k]=count1/count[1]


		probno=prob[0]
		probyes=prob[1]
		for i in range(test.shape[1]-1):
			probno=probno*prob0[i]
			probyes=probyes*prob1[i]


		if probno>probyes:
			predict='no'
		else:
			predict='yes'

		print(t+1,"\t",predict,"\t	",test[t,test.shape[1]-1])
		if predict== test[t,test.shape[1]-1]:
			accuracy+=1  #count number of yes's or no's (predict)

	final_accuracy=(accuracy/test.shape[0])*100
	print("accuracy",final_accuracy,"%")
	pdb.set_trace()
	

metadata,traindata= read_data("tennis.csv")
splitRatio=0.6
trainingset, testset=splitDataset(traindata, splitRatio)
training=np.array(trainingset)
testing=np.array(testset)
print("\n\n_________Training Data_____________")
print(trainingset)
print("_________Test Data_____________")
print(testset)

classify(training,testing)
