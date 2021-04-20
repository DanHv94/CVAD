import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.feature_selection import SelectFromModel

def load_features(filepath):
	X = list()  
	y = list()
	with open(filepath) as archivo:
		for linea in archivo:
			data = linea.rstrip("\n").split(",")
			X.append(data[:len(data) - 1])
			y.append(data[len(data) - 1])
	#print(X)
	return np.array(X,dtype="float"), np.array(y,dtype="int16")




X_MEXA3T_train, y_MEXA3T_train = load_features("..\\Corpus\\MEX-A3T\\MEX_A3T_train_features_v2_3.txt")
'''param_grid = {'C': [0.01,0.1,1, 10, 100, 1000], "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]} 
grid = GridSearchCV(LogisticRegression(max_iter = 500),param_grid,refit="f1_macro",verbose=3,scoring=["f1_macro", "f1_micro","f1","accuracy"])
grid.fit(X_MEXA3T_train, y_MEXA3T_train)'''
print("3 MEX-A3T")
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001],"kernel": ["linear"]} 
grid = GridSearchCV(svm.SVC(),param_grid,refit="f1_macro",verbose=3,scoring=["f1","f1_micro","f1_macro","recall","accuracy"])
grid.fit(X_MEXA3T_train, y_MEXA3T_train)

print("3) Mejores parámetros d:",grid.best_params_)
print("Mejor resutado:",grid.best_score_)