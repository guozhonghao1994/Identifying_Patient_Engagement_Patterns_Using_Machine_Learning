# preprocess
import numpy as np
import csv

with open("train.csv",mode='r',encoding='utf-8',newline='') as f:
	data = csv.reader(f)
	matrix1 = []
	for row in data:
		matrix1.append(row)
	matrix1 = np.array(matrix1)
	# label1 = matrix1[0][1:]

with open("Enrollment Data (collated final).csv",mode='r',encoding='utf-8',newline='') as f:
	# drive里面更新了Enrollment Data (collated final).csv，手动删除了一个重复病人
	data = csv.reader(f)
	matrix2 = []
	for row in data:
		matrix2.append(row)
	matrix2 = np.array(matrix2)
	item2 = matrix2[0]
	# print(item2)

patient_id1 = matrix1[1:,2]
patient_id2 = matrix2[1:,1]

for row_num,id_num in enumerate(patient_id2):
	if id_num in patient_id1:
		item2 = np.concatenate((item2,matrix2[row_num+1,:]),axis=0)
matrix2 = item2.reshape(-1,8)
matrix1 = np.delete(matrix1,0,axis=1)
matrix2 = matrix2[matrix2[:,1].argsort()][::-1]
matrix1 = matrix1[matrix1[:,1].argsort()][::-1]

y = matrix1[:,-1].reshape(-1,1)
# print(y.shape)
training_data = np.concatenate((matrix2,y),axis=1)
print(training_data.shape)

with open('training_data.csv','w',newline='') as w:
	writer = csv.writer(w)
	for row_data in training_data:
		writer.writerow(row_data)
	w.close()

np.save('training_data.npy',training_data)
print("preprocess done!")






