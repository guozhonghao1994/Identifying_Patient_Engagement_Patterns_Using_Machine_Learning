# Notification
import csv
import heapq
import numpy as np


def read_data(filename):
	data_clean = np.load(filename)
	sample_num = data_clean.shape[0]
	feature_num = data_clean.shape[1]
	return data_clean,sample_num,feature_num


def read_notification(filename):
	with open(filename,mode='r',encoding='utf8',newline='') as f:
		data = csv.reader(f)
		matrix = []
		for row in data:
			matrix.append(row)
		matrix = np.array(matrix[1:]).astype(int)
		note_list = np.unique(matrix[:,1])
		f.close()
	return matrix,note_list


def create_dict(notification_data):
	dict = {}
	for i in np.unique(notification_data[:,0]):
		dict[i] = [] 
	for row_num,patient_id in enumerate(notification_data[:,0]):
		dict[patient_id].append(notification_data[row_num,1])
	return dict


def create_notification_matrix(dataset_clean,id_note_dict,notification_list):
	patient_id_list = id_note_dict.keys()
	matrix = np.zeros((dataset_clean[1:].shape[0],len(patient_id_list)))
	for i,patient_id in enumerate(dataset_clean[1:,0]):
		if int(patient_id) not in patient_id_list:
			continue
		else:
			for j,notification_id in enumerate(notification_list):
				if notification_id in id_note_dict[int(patient_id)]:
					matrix[i][j] = 1
	return matrix


def dim_reduction(training_note,testing_note):
	note = np.vstack((training_note,testing_note))
	sum_list = np.sum(note,axis=0)
	first_30 = np.unique(heapq.nlargest(30,sum_list))
	column_num = np.where([sum_list == k for k in first_30])[1]
	print("totally",len(column_num),"messages are chosen.")
	return column_num


def generate_new_data(data_clean,note_matrix,column_num):
	temp = data_clean[1:,:-1]
	for col in column_num:
		temp = np.hstack((temp,note_matrix[:,col].reshape(-1,1)))
	temp = np.hstack((temp,data_clean[1:,-1].reshape(-1,1)))
	assert(temp.shape == (data_clean.shape[0]-1,data_clean.shape[1]+len(column_num)))
	return temp


def save_data(filename,data):
	np.save(filename,data)


def main():
	training_data_clean,training_num,training_feature = read_data('training_data_clean.npy')
	testing_data_clean,testing_num,testing_feature = read_data('testing_data_clean.npy')
	notification_data,notification_list = read_notification('notification.csv') 
	id_note_dict = create_dict(notification_data) # call notification list of a patient by "id_note_dict[patient_id]"
	training_note = create_notification_matrix(training_data_clean,id_note_dict,notification_list)
	testing_note = create_notification_matrix(testing_data_clean,id_note_dict,notification_list)
	first_30_note = dim_reduction(training_note,testing_note)
	final_training = generate_new_data(training_data_clean,training_note,first_30_note)
	final_testing = generate_new_data(testing_data_clean,testing_note,first_30_note)
	save_data('train_with_message',final_training)
	save_data('test_with_message',final_testing)
	print("data saved!")


if __name__ == '__main__':
	main()