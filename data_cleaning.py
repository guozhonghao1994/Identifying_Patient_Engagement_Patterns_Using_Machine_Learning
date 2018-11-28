# data cleaning
import numpy as np
import datetime
from data_preprocess import write_csv

# load npy
training_data = np.load('training_data.npy')
testing_data = np.load('testing_data.npy')
print("data loaded")

# get the training dataset with full features
gender_col = int(np.argwhere(training_data[0] == 'Gender'))
birthday_col = int(np.argwhere(training_data[0] == 'Date of Birth'))
training_data_full = training_data[0]
for row_num in range(1,training_data.shape[0]):
	if training_data[row_num][gender_col] and training_data[row_num][birthday_col]:
		training_data_full = np.concatenate((training_data_full,training_data[row_num]),axis=0)

training_data = training_data_full.reshape(-1,9)
assert(training_data.shape == (965,9))

# get the testing dataset with full features
gender_col = int(np.argwhere(testing_data[0] == 'Gender'))
birthday_col = int(np.argwhere(testing_data[0] == 'Date of Birth'))
testing_data_full = testing_data[0]
for row_num in range(1,testing_data.shape[0]):
	if testing_data[row_num][gender_col] and testing_data[row_num][birthday_col]:
		testing_data_full = np.concatenate((testing_data_full,testing_data[row_num]),axis=0)

testing_data = testing_data_full.reshape(-1,9)
assert(testing_data.shape == (209,9))

# change all bool features to binary 0/1
# TRUE = 1; FALSE = 0; Male = 1; Female = 0
training_data[training_data == 'TRUE'] = 1
training_data[training_data == 'FALSE'] = 0
training_data[training_data == 'Male'] = 1
training_data[training_data == 'Female'] = 0

testing_data[testing_data == 'TRUE'] = 1
testing_data[testing_data == 'FALSE'] = 0
testing_data[testing_data == 'Male'] = 1
testing_data[testing_data == 'Female'] = 0
print("bool to binary done")

# switch date to time period
# waiting time(days) = procedure date - registration data
# age(days) = procedure date - date of birth

# training
Waiting_Time = ['Waiting Time']
Age = ['Age']
for row_num in range(1,training_data.shape[0]):
	reg_time = training_data[row_num][2]
	pro_time = training_data[row_num][3]
	birth_time = training_data[row_num][6]
	reg_date = datetime.datetime(int(reg_time.split('/')[0]),int(reg_time.split('/')[1]),int(reg_time.split('/')[2]))
	pro_date = datetime.datetime(int(pro_time.split('/')[0]),int(pro_time.split('/')[1]),int(pro_time.split('/')[2]))
	birth_date = datetime.datetime(int(birth_time.split('/')[0]),int(birth_time.split('/')[1]),int(birth_time.split('/')[2]))
	waiting_time = abs((pro_date - reg_date).days)
	Waiting_Time.append(waiting_time)
	age = (pro_date - birth_date).days
	Age.append(age)
# print(Waiting_Time,Age)
Waiting_Time = np.array(Waiting_Time)
Age = np.array(Age)
# print(Waiting_Time.shape,Age.shape)
training_data[:,3] = Waiting_Time
training_data[:,6] = Age
training_data = np.delete(training_data,2,axis=1)
assert(training_data.shape == (965,8))

# testing
Waiting_Time = ['Waiting Time']
Age = ['Age']
for row_num in range(1,testing_data.shape[0]):
	reg_time = testing_data[row_num][2]
	pro_time = testing_data[row_num][3]
	birth_time = testing_data[row_num][6]
	reg_date = datetime.datetime(int(reg_time.split('/')[0]),int(reg_time.split('/')[1]),int(reg_time.split('/')[2]))
	pro_date = datetime.datetime(int(pro_time.split('/')[0]),int(pro_time.split('/')[1]),int(pro_time.split('/')[2]))
	birth_date = datetime.datetime(int(birth_time.split('/')[0]),int(birth_time.split('/')[1]),int(birth_time.split('/')[2]))
	waiting_time = abs((pro_date - reg_date).days)
	Waiting_Time.append(waiting_time)
	age = (pro_date - birth_date).days
	Age.append(age)
# print(Waiting_Time,Age)
Waiting_Time = np.array(Waiting_Time)
Age = np.array(Age)
# print(Waiting_Time.shape,Age.shape)
testing_data[:,3] = Waiting_Time
testing_data[:,6] = Age
testing_data = np.delete(testing_data,2,axis=1)
assert(testing_data.shape == (209,8))

# save as npy, csv
write_csv(training_data,'training_data_clean.csv')
write_csv(testing_data,'testing_data_clean.csv')

np.save('training_data_clean.npy',training_data)
np.save('testing_data_clean.npy',testing_data)






