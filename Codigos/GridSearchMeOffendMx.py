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
	return np.array(X,dtype="float"), np.array(y,dtype="int16")

X_MeOffendEsMx_train, y_MeOffendEsMx_train = load_features("..\\Corpus\\MeOffendEs\\OffendMex\\MeOffendEsMx_train_features.txt")
'''
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001],"kernel": ["linear","poly"]} 
grid = GridSearchCV(svm.SVC(),param_grid,refit="f1",verbose=3,scoring=["f1","f1_micro","f1_macro","recall","accuracy"])
grid.fit(X_MeOffendEsMx_train[:2500], y_MeOffendEsMx_train[:2500])

print("Mejores parámetros:",grid.best_params_)
print("Mejor resutado:",grid.best_score_)'''
'''
param_grid = {'C': [0.01,0.1,1, 10, 100, 1000], "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]} 
grid = GridSearchCV(LogisticRegression(max_iter = 500),param_grid,refit="f1_macro",verbose=3,scoring=["f1_macro", "f1_micro","f1","accuracy"])
grid.fit(X_MeOffendEsMx_train, y_MeOffendEsMx_train)

print("Mejores parámetros:",grid.best_params_)
print("Mejor resutado:",grid.best_score_)'''

X_MEXA3T_train, y_MEXA3T_train = load_features("..\\Corpus\\MEX-A3T\\MEX_A3T_train_features.txt")
'''
model = LogisticRegression(C=1, solver="saga", max_iter=500)
model.fit(X_MEXA3T_train, y_MEXA3T_train)
MEXA3T_predicts = model.predict(X_MEXA3T_train)
print(classification_report(y_MEXA3T_train, MEXA3T_predicts))'''

'''
model = LogisticRegression(C=1, solver="saga", max_iter=500)
model.fit(X_MEXA3T_train, y_MEXA3T_train)
MEXA3T_predicts = model.predict(X_MEXA3T_train)
print(classification_report(y_MEXA3T_train, MEXA3T_predicts))
print(model.coef_)'''

lrl = LogisticRegression(C=1, solver="saga", penalty="l1", max_iter=500).fit(X_MEXA3T_train, y_MEXA3T_train)
model = SelectFromModel(lrl, prefit=True)
X_MEXA3T_train_new = model.transform(X_MEXA3T_train)

#print(X_MEXA3T_train.shape)
#print(X_MEXA3T_train[0])
#print(X_MEXA3T_train_new.shape)
#print(X_MEXA3T_train_new[0])
print("MEXA3T")
print(model.get_support(True))


lrl = LogisticRegression(C=1, solver="sag").fit(X_MeOffendEsMx_train, y_MeOffendEsMx_train)
model = SelectFromModel(lrl, prefit=True)
#X_MEXA3T_train_new = model.transform(X_MEXA3T_train)

print("MeOffendMx")
print(model.get_support(True))
