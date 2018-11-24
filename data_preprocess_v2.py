# preprocess
import csv
import numpy as np

def read_csv(filename):
	with open(filename,mode='r',encoding='utf8',newline='') as f:
		data = csv.reader(f)
		matrix = []
		for row in data:
			matrix.append(row)
		matrix = np.array(matrix)
		label = matrix[0]
		f.close()
	return matrix,label


def write_csv(csv_data,filename):
	with open(filename,'w',newline='') as w:
		writer = csv.writer(w)
		for row in csv_data:
			writer.writerow(row)
		w.close()
	print(filename + ' done!')


def combine(enrollid,combineid,item,combine_matrix):
	for row_num,id_num in enumerate(enrollid):
		if id_num in combineid:
			item = np.concatenate((item,combine_matrix[row_num+1,:]),axis=0)
		combined = item.reshape(-1,8)
		combined = combined[combined[:,1].argsort()][::-1]
	return combined


def main():
	filename = ['Enrollment Data (collated final).csv',
				'train.csv','test.csv']

	enroll_matrix, enroll_label = read_csv(filename[0])
	train_matrix, train_label = read_csv(filename[1])
	test_matrix, test_label = read_csv(filename[2])

	enroll_id = enroll_matrix[1:,1]
	train_id = train_matrix[1:,2]
	test_id = test_matrix[1:,2]

	train_combined = combine(enrollid=enroll_id,combineid=train_id,item=enroll_label,combine_matrix=enroll_matrix)
	test_combined = combine(enrollid=enroll_id,combineid=test_id,item=enroll_label,combine_matrix=enroll_matrix)

	train_matrix = np.delete(train_matrix,0,axis=1)
	train_matrix = train_matrix[train_matrix[:,1].argsort()][::-1]
	test_matrix = np.delete(test_matrix,0,axis=1)
	test_matrix = test_matrix[test_matrix[:,1].argsort()][::-1]

	train_label = train_matrix[:,-1].reshape(-1,1)
	test_label = test_matrix[:,-1].reshape(-1,1)

	training_data = np.concatenate((train_combined,train_label),axis=1)
	testing_data = np.concatenate((test_combined,test_label),axis=1)
	print(training_data.shape,testing_data.shape)

	# saving csv
	# write_csv(training_data,'training_data.csv')
	# write_csv(testing_data,'testing_data.csv')

	# saving npy
	np.save('training_data.npy',training_data)
	np.save('testing_data.npy',testing_data)


if __name__ == "__main__":
	main()
