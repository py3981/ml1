import csv
a = []
print("\n The Given Training Data Set \n")
with open('ws.csv', 'r') as csvFile:
  reader = csv.reader(csvFile)
  for row in reader:
	a.append (row)
	print(row)
num_attributes=len(a[0])-1
print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)
# Comparing with First Training Example
for j in range(0,num_attributes):
  hypothesis[j] = a[0][j];
# Comparing with Remaining Training Examples of Given Data Set
print("\n Find S: Finding a Maximally Specific Hypothesis\n")
for i in range(0,len(a)):
   if a[i][num_attributes]=='Yes':
      for j in range(0,num_attributes):
         if a[i][j]!=hypothesis[j]:
            hypothesis[j]='?'
   print("For training example No",(i+1),"hypothesis is", hypothesis)
print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)
