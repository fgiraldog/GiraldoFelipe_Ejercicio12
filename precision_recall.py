import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics as skm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Importacion de datos
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#PCA
def PCA(x):
	cov = np.cov(x.T)
	valores, vectores = np.linalg.eig(cov)
	valores = np.real(valores)
	vectores = np.real(vectores)
	ii = np.argsort(-valores)
	valores = valores[ii]
	vectores = vectores[:,ii]

	return vectores

numero = 1
dd = y_train == numero
vectores_ones = PCA(x_train[dd])

numero = 1
ii = y_train != numero
vectores_others = PCA(x_train[ii])

vectores = PCA(x_train)

x_train_ones = x_train @ vectores_ones
x_test_ones = x_test @ vectores_ones

x_train_others = x_train @ vectores_others
x_test_others = x_test @ vectores_others

x_train_all = x_train @ vectores
x_test_all = x_test @ vectores

y_train[y_train != 1] = 0
y_test[y_test != 1] = 0

def prob_pre_recall(x_fit,x,y_fit,y):

	lda = LinearDiscriminantAnalysis()
	lda.fit(x_fit[:,0:10],y_fit)
	proba = lda.predict_proba(x[:,0:10])[:,1]
	precision, recall, threshold = skm.precision_recall_curve(y,proba,pos_label = 1)
	f1 = 2*precision*recall/(precision+recall)

	return precision, recall, threshold, f1

p_ones, r_ones, t_ones, f1_ones = prob_pre_recall(x_train_ones,x_test_ones,y_train,y_test)
p_others, r_others, t_others, f1_others = prob_pre_recall(x_train_others,x_test_others,y_train,y_test)
p_all, r_all, t_all, f1_all = prob_pre_recall(x_train_all,x_test_all,y_train,y_test)

ones_max = np.argmax(f1_ones[1:])
others_max = np.argmax(f1_others[1:])
all_max = np.argmax(f1_all[1:])

plt.figure(figsize = (10,5))
plt.subplot(121)
plt.scatter(r_ones[ones_max], p_ones[ones_max] , c ='c')
plt.scatter(r_others[others_max], p_others[others_max], c ='c')
plt.scatter(r_all[all_max], p_all[all_max], c ='c')
plt.plot(r_ones, p_ones,label = 'Ones')
plt.plot(r_others, p_others,label = 'Others')
plt.plot(r_all, p_all,label = 'All')
plt.xlabel('Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.subplot(122)
plt.plot(t_ones,f1_ones[1:],label = 'Ones')
plt.plot(t_others,f1_others[1:],label = 'Others')
plt.plot(t_all,f1_all[1:],label = 'All')
plt.scatter(t_ones[ones_max], f1_ones[1:][ones_max] , c ='c')
plt.scatter(t_others[others_max], f1_others[1:][others_max], c ='c')
plt.scatter(t_all[all_max], f1_all[1:][all_max], c ='c')
plt.xlabel('Probability')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig('F1_prec_recall.png')

