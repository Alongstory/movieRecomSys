#__Author__ = 'zhangxl'
# 04/05/2017
import numpy as np 
import pandas as pd 
from sklearn import svm
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sn
import preprocess as pr

train_df = pd.DataFrame.from_csv('train.csv')

"""Analysis for Total Distribution"""
def analy_total():
	y = train_df.Survived.value_counts()
	x = [0, 1]
	sum = y[0] + y[1]
	plt.bar(x[0], y[0]*1.0/sum, width=0.3, color='r',label='Perished',align='center')
	plt.bar(x[1], y[1]*1.0/sum, width=0.3, color='b',label='Survived',align='center')
	plt.xticks(x)
	plt.xlabel('Label', fontsize=10)
	plt.ylabel('Rate of Survivors', fontsize=10)
	plt.title("Total Distribution of Survivors", fontsize=10, fontweight = 'bold')
	plt.legend()
	plt.show()

""" Analysis for Pclass"""
def analy_pclass():
	plt.subplot(1,3,1)
	x = [0, 1]
	y = {}
	for i in range(0,3):
		y[i] =train_df.Survived[(train_df.Pclass == (i+1))].value_counts()
	sum = y[0][0] + y[0][1]
	plt.bar(x[0], y[0][0]*1.0/sum, width = 0.3, color = 'r', label = 'Perished',align='center')
	plt.bar(x[1], y[0][1]*1.0/sum, width = 0.3, color = 'b', label = 'Survived',align='center')
	plt.xticks(x)
	plt.legend()
	plt.xlabel('Label', fontsize=10)
	plt.ylabel('Rate of Survivors', fontsize=10)
	plt.title("Distribution of Survivors in Class 1", fontsize=10, fontweight = 'bold')

	plt.subplot(1,3,2)
	sum = y[1][0] + y[1][1]
	plt.bar(x[0], y[1][0]*1.0/sum, width = 0.3, color = 'r', label = 'Perished',align='center')
	plt.bar(x[1], y[1][1]*1.0/sum, width = 0.3, color = 'b', label = 'Survived',align='center')
	plt.xticks(x)
	plt.legend()
	plt.xlabel('Label', fontsize=10)
	plt.title("Distribution of Survivors in Class 2", fontsize=10, fontweight = 'bold')

	plt.subplot(1,3,3)
	sum = y[2][0] + y[2][1]
	plt.bar(x[0], y[2][0]*1.0/sum, width = 0.3, color = 'r', label = 'Perished',align='center')
	plt.bar(x[1], y[2][1]*1.0/sum, width = 0.3, color = 'b', label = 'Survived',align='center')
	plt.xticks(x)
	plt.legend()
	plt.xlabel('Label', fontsize=10)
	plt.title("Distribution of Survivors in Class 3", fontsize=10, fontweight = 'bold')
	plt.show()

"""Analysis for NameLength"""
def analy_name():
	train_name = {}
	y = Counter()
	train_df.NameLength = train_df.Name.apply(lambda x:len(x))

	#print train_df.NameLength==12
	for i in range(0,2):
		train_name[i] = train_df.NameLength[train_df.Survived == i].value_counts().sort_index()
	train_name[0].plot.bar(color='r', alpha=0.4, label = 'Perished', rot = 0, fontsize=8)
	train_name[1].plot.bar(color='b', alpha=0.4, label = 'Survived', rot = 0, fontsize=8)
	
	plt.title("Survival Compared by NameLength", fontweight = "bold")
	plt.legend()
	plt.show()

"""Analysis for Sex"""
def analy_sex():
	train_sex = {}
	for i in range(0,2):
		train_sex[i] = train_df.Sex[train_df.Survived == i].value_counts().sort_index()
	train_sex[0].plot.bar(color='r', alpha=0.4, label = 'Perished', rot = 0, fontsize=8)
	train_sex[1].plot.bar(color='b', alpha=0.4, label = 'Survived', rot = 0, fontsize=8)
	
	plt.title("Survival Compared by Sex", fontweight = "bold")
	plt.legend()
	plt.show()	

"""Analysis for Age"""	
def analy_age():
	f = train_df[np.isfinite(train_df['Age'])]
	f.groupby('Survived').Age.plot(kind='kde')
	train_age = {}
	age_mean = train_df.Age.mean()
	train_df['Age'] = train_df['Age'].fillna(age_mean)
	for i in range(0,2):
		train_age[i] = train_df.Age[train_df.Survived == i]
	sn.kdeplot(train_age[0], shade = True, color = 'r', label = 'Perished')
	sn.kdeplot(train_age[1], shade = True, color = 'b', label = 'Survived')
	plt.title("Survival Compared by Age", fontweight = "bold")
	plt.legend()
	plt.show()	

"""Analysis for SibSp&Parch as Family"""
def analy_fam():
	plt.subplot(1,3,1)
	train_fam = {}
	train_df.FamilySize = train_df.SibSp #+ train_df.Parch
	for i in range(0,2):
		train_fam[i] = train_df.FamilySize[train_df.Survived == i].value_counts().sort_index()
	train_fam[0].plot.bar(color='r', alpha=0.4, label = 'Perished', rot = 0, fontsize=8)
	train_fam[1].plot.bar(color='b', alpha=0.4, label = 'Survived', rot = 0, fontsize=8)
	plt.title("Survival Compared by SibSp", fontweight = "bold")
	plt.legend()

	plt.subplot(1,3,2)
	train_fam = {}
	train_df.FamilySize = train_df.Parch
	for i in range(0,2):
		train_fam[i] = train_df.FamilySize[train_df.Survived == i].value_counts().sort_index()
	train_fam[0].plot.bar(color='r', alpha=0.4, label = 'Perished', rot = 0, fontsize=8)
	train_fam[1].plot.bar(color='b', alpha=0.4, label = 'Survived', rot = 0, fontsize=8)
	plt.title("Survival Compared by Parch", fontweight = "bold")
	plt.legend()

	plt.subplot(1,3,3)
	train_fam = {}
	train_df.FamilySize = train_df.SibSp + train_df.Parch
	for i in range(0,2):
		train_fam[i] = train_df.FamilySize[train_df.Survived == i].value_counts().sort_index()
	train_fam[0].plot.bar(color='r', alpha=0.4, label = 'Perished', rot = 0, fontsize=8)
	train_fam[1].plot.bar(color='b', alpha=0.4, label = 'Survived', rot = 0, fontsize=8)
	plt.title("Survival Compared by FamilySize", fontweight = "bold")
	plt.legend()
	plt.show()	

"""Analysis for Fare"""
def analy_fare():
	train_fare = {}
	for i in range(0,2):
		train_fare[i] = train_df.Fare[train_df.Survived == i]
	sn.kdeplot(train_fare[0], shade = True, color = 'r', label = 'Perished')
	sn.kdeplot(train_fare[1], shade = True, color = 'b', label = 'Survived')
	plt.title("Survival Compared by Fare", fontweight = "bold")
	plt.legend()
	plt.show()

"""Analysis for Cabin"""
def analy_cab():
	train_df['CabinNo'] = pd.Categorical(train_df.Cabin.fillna('0').apply(lambda x: x[0])).codes
	sn.kdeplot(train_df.CabinNo[(train_df.Survived==0)&(train_df.CabinNo!=0)],shade=True,color='r',label='Perished')
	sn.kdeplot(train_df.CabinNo[(train_df.Survived==1)&(train_df.CabinNo!=0)],shade=True,color='b',label='Survived')
	plt.title("Survival Compared by Cabin", fontweight = "bold")
	plt.legend()
	plt.show()	

"""Analysis for Embarked"""
def analy_embark():
	x = [0, 1]
	y = {}
	y[0] = train_df.Survived[train_df.Embarked =='C'].value_counts()
	y[1] = train_df.Survived[train_df.Embarked =='S'].value_counts()
	y[2] = train_df.Survived[train_df.Embarked =='Q'].value_counts()
	plt.subplot(1,3,1)
	sum = y[0][0] + y[0][1]
	plt.bar(x[0], y[0][0]*1.0/sum, width = 0.3, color = 'r', label = 'Perished', align='center')
	plt.bar(x[1], y[0][1]*1.0/sum, width = 0.3, color = 'b', label = 'Survived', align='center')
	plt.xticks(x)
	plt.legend()
	plt.xlabel('Label', fontsize=10)
	plt.ylabel('Rate of Survivors', fontsize=10)
	plt.title("Distribution of Survivors in Embarkation C", fontsize=10, fontweight = 'bold')

	plt.subplot(1,3,2)
	sum = y[1][0] + y[1][1]
	plt.bar(x[0], y[1][0]*1.0/sum, width = 0.3, color = 'r', label = 'Perished', align='center')
	plt.bar(x[1], y[1][1]*1.0/sum, width = 0.3, color = 'b', label = 'Survived', align='center')
	plt.xticks(x)
	plt.legend()
	plt.xlabel('Label', fontsize=10)
	plt.title("Distribution of Survivors in Embarkation S", fontsize=10, fontweight = 'bold')

	plt.subplot(1,3,3)
	sum = y[2][0] + y[2][1]
	plt.bar(x[0], y[2][0]*1.0/sum, width = 0.3, color = 'r', label = 'Perished', align='center')
	plt.bar(x[1], y[2][1]*1.0/sum, width = 0.3, color = 'b', label = 'Survived', align='center')
	plt.xticks(x)
	plt.legend()
	plt.xlabel('Label', fontsize=10)
	plt.title("Distribution of Survivors in Embarkation Q", fontsize=10, fontweight = 'bold')
	plt.show()


'''Analysis for correlation'''
def data_corr():
	# Get data according to analysis above
	train_path = pr.process_data('train.csv')
	train_data = pd.DataFrame.from_csv(train_path)

	sn.set(style="white")
	sn.set(font_scale=0.7)
    # Compute the correlation matrix
	corr = train_data.corr()
    # Generate a mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
	cmap = sn.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
	sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
	plt.show()

def analysis_data():
	analy_sex()
	analy_age()
	analy_fam()
	analy_fare()
	analy_cab()
	analy_embark()

if __name__ == '__main__':
	analysis_data()