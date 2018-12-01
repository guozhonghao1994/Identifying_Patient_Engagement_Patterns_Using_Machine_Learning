# SVC
import numpy as np
import time
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


training_data = np.load('training_data_clean.npy')
testing_data = np.load('testing_data_clean.npy')
print("training samples:",training_data.shape[0]-1,"testing samples:",testing_data.shape[0]-1)
print("-------------------------------------")

# scale data
training_sample = training_data[1:,1:-1].astype(float)
training_sample = preprocessing.scale(training_sample)
training_label = training_data[1:,-1].astype(int)

testing_sample = testing_data[1:,1:-1].astype(float)
testing_sample = preprocessing.scale(testing_sample)
testing_label = testing_data[1:,-1].astype(int)

# feature importance
model = ExtraTreesClassifier()
model.fit(training_sample, training_label)
print("Relative importance of each attribute")
print(training_data[0][1:-1],'\n',model.feature_importances_)
print("-------------------------------------")

# training & grid search
def train():
	local_time = str(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
	param = {'kernel':('linear','poly','rbf'),'C':range(1,5), 'gamma':[0.125, 0.25, 0.5 ,1],'class_weight':['balanced']}
	svr = svm.SVC()
	clf = GridSearchCV(svr,param)
	clf.fit(training_sample,training_label)
	# optimal params
	with open('cv_results_' + local_time + '.txt','w') as f:
		f.write(str(clf.cv_results_))
		f.close()
	print("best score:",clf.best_score_)
	print("best estimator:",clf.best_estimator_)
	# save model
	joblib.dump(clf.best_estimator_,'SVC_model.m')
	print("model saved!")

# test
def test():
	predictor = joblib.load('SVC_model.m')
	print("model loaded!")
	print(predictor.predict(testing_sample))
	testing_label_0 = np.zeros(testing_data.shape[0]-1).reshape(-1,1)
	print("accuracy:",predictor.score(testing_sample,testing_label_0))

MODE = 0
if MODE == 1:
	train()
else:
	test()


