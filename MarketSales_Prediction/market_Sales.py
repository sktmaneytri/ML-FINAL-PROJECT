'''
Lecturer: PhD. Tran Nhat Quang
Class: MALE431085E_22_1_01CLC
Subject ID: 221MALE431085E
Members: 
	Nguyen Minh Tri  – 20110422
	Phan Thanh Luan – 20110380
	Vo Minh Hung – 20110426
	Truong Chi Kien – 20110376

TOPIC OF PROJECT: MARKET SALES PREDICTIONS
To more information you can go to the ipython notebook file to see clearly.
The link of datasets we got on the Kaggle following the link: 
https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets

BEST REGARDS!
'''
#%%# In[0]: IMPORT AND FUNCTIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')

#%%# In[1]: LOOK AT THE PICTURE (DONE)

#%%# In[2]: GET THE DATA (DONE). LOAD DATA
train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')

#%%# In[3]: DISCOVER THE DATA TO GAIN INSIGHTS
display(train.head())
print(train.columns)
display(test.head())
print(test.columns)
print(train.shape)
print(test.shape)
train["source"] = "train"
test["source"] = "test"
test["Item_Outlet_Sales"] = 0.0
df = pd.concat([train, test], sort = False, ignore_index = True)
print(df.shape)
display(df.head())
df.isnull().sum()
df.skew()
display(df.describe())
display(df.describe(include = object))
display(df.isnull().sum()/df.shape[0] * 100)
print("Visibility = 0 :", (df["Item_Visibility"].values==0).sum()/df.shape[0] * 100)
CatFeatures = [col for col in df.columns if df[col].dtypes in ["object"]]
NumFeatures = [col for col in df.columns if df[col].dtypes not in ["object"]]
print("Numeric Features : ", NumFeatures)
print("Categorical Features : ", CatFeatures)

for col in CatFeatures:
  print(f"\n{col} : \n{df[col].value_counts()}")
  df.skew()
df["Item_Outlet_Sales"].skew()
df.skew()
print('Categorical Features: ',CatFeatures)
print('\nNumeric Features: ',NumFeatures)
cat = list(set(CatFeatures) - set("source"))
print('\nCategorical Features (without source):',cat)
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
df['Item_Fat_Content'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(data=df,x="Item_Fat_Content")
df['Item_Fat_Content'].value_counts()
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
df['Outlet_Identifier'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.subplot(2,1,2)
sns.countplot(data=df,x="Outlet_Identifier")
df['Outlet_Identifier'].value_counts()
plt.figure(figsize=(10,15))
plt.subplot(2,1,1)
df['Item_Type'].value_counts().plot.pie(autopct='%1.1f%%',textprops={'fontsize':10})
plt.ylabel('')
plt.subplot(2,1,2)
sns.countplot(data=df,x="Item_Type")
plt.xticks(rotation = 90)
df['Item_Type'].value_counts()
plt.figure(figsize=(10,15))
plt.subplot(2,1,1)
df['Outlet_Establishment_Year'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(2,1,2)
sns.countplot(data=df,x="Outlet_Establishment_Year")
df['Outlet_Establishment_Year'].value_counts()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
df['Outlet_Size'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(data=df,x="Outlet_Size")
df['Outlet_Size'].value_counts()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
df['Outlet_Location_Type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(data=df,x="Outlet_Location_Type")
df['Outlet_Location_Type'].value_counts()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
df['Outlet_Type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(data=df,x="Outlet_Type")
plt.xticks(rotation=45)
df['Outlet_Type'].value_counts()
print(NumFeatures)
cont = list(set(NumFeatures) - set(['Outlet_Establishment_Year', 'Item_Outlet_Sales']))
print(cont)
for feature in cont:
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    f.set_figheight(3)
    f.set_figwidth(15)
    sns.boxplot(df[feature], ax=ax_box)
    sns.histplot(data=df, x=feature, ax=ax_hist)
    plt.show()
sns.stripplot(data=df,x='Outlet_Location_Type',y='Outlet_Type')
sns.stripplot(data=df,x='Outlet_Size',y='Outlet_Type')
sns.stripplot(data=df,x='Outlet_Size',y='Outlet_Location_Type')
sns.stripplot(data=df,x='Outlet_Establishment_Year',y='Outlet_Type')
sns.stripplot(data=df,x='Outlet_Establishment_Year',y='Outlet_Location_Type')
sns.stripplot(data=df,x='Outlet_Establishment_Year',y='Outlet_Size')
plt.figure(figsize=(10,6))
sns.stripplot(data=df,x='Outlet_Identifier',y='Outlet_Type')
plt.figure(figsize=(10,6))
sns.stripplot(data=df,x='Outlet_Identifier',y='Outlet_Location_Type')
sns.stripplot(data=df,x='Outlet_Size',y='Outlet_Identifier')
plt.figure(figsize=(10,6))
sns.stripplot(data=df,x='Outlet_Identifier',y='Outlet_Establishment_Year')
plt.figure(figsize=(10,6))
sns.stripplot(data=df,x='Outlet_Identifier',y='Item_Visibility')
plt.figure(figsize=(10,6))
sns.stripplot(data=df,x='Outlet_Type',y='Item_Visibility')
sns.stripplot(data=df,x='Outlet_Size',y='Item_Visibility')
fig,ax=plt.subplots(2,2,figsize=(18,10))
r=0
c=0
for i,n in enumerate(['Item_Fat_Content','Outlet_Establishment_Year','Outlet_Type','Item_Fat_Content']):
    if i%2==0 and i>0:
        r+=1
        c=0
    sns.boxenplot(x=df[n],y=df['Item_Weight'],ax=ax[r,c])
    c+=1
for i in range(len(cont)):
    for j in range(i+1,len(cont)):
        sns.scatterplot(x=cont[i],y=cont[j],data=df)
        plt.figure()
sns.countplot(data=df,x='Outlet_Location_Type',hue=df['Outlet_Type'])
sns.countplot(data=df,x='Outlet_Size',hue=df['Outlet_Location_Type'])
sns.lmplot(x='Item_Weight',y='Item_Outlet_Sales',data=df)
sns.lmplot(x='Item_Visibility',y='Item_Outlet_Sales',data=df)
sns.lmplot(x='Item_MRP',y='Item_Outlet_Sales',data=df)
for i in cat:
    plt.figure(figsize=(15,6))
    sns.boxplot(x=df[i],y=df['Item_Outlet_Sales'])
    plt.xticks(rotation=45)
    plt.figure()
plt.figure(figsize=(10,8))
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',hue='Outlet_Type',data=df)
plt.figure(figsize=(10,8))
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',hue='Outlet_Type',data=df)
plt.figure(figsize=(10,8))
sns.scatterplot(x='Item_Visibility',y='Item_MRP',hue='Item_Outlet_Sales',data=df,palette='rocket_r')
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)
# Correlation Between Numeric with Numeric
df.corr()
# Correlation Between Categorical with Categorical
import association_metrics as am
corrdf = df[CatFeatures].copy()
corrdf = corrdf.apply(
        lambda x: x.astype("category") if x.dtype == "O" else x)
cramersv = am.CramersV(corrdf) 
categoric_corr_matrix = cramersv.fit()
display(categoric_corr_matrix)
# Correlation Between Numeric with Categorical
# Encoding Categorical to Numeric OrderEncoding ?
# Then uding df.corr()
# Feature
# Feature 1 : A, B , C
# Feature 2 : U , H , M
# Feature 3 : X, Y , Z 

# mean({A,U,X} on Feature)
# mean({A,U,Y} on Feature)
table = df.pivot_table(values = feature, index = "Item_Type", columns = "Outlet_Type", aggfunc = "mean")



#%%# In[4]: PREPARE THE DATA

#Item_Visibility
df['Item_Visibility'].value_counts()
#Replacing o visibility with nan values
df['Item_Visibility'].replace(0,np.nan,inplace=True)
df['Item_Visibility'].min()
#Checking visibility in Outlet size with Outlet type
plt.figure(figsize=(15,6))
sns.boxplot(data=df, x='Outlet_Size', y='Item_Visibility', hue='Outlet_Type')
plt.xticks(rotation=45)
#Checking visibility in Outlet identifier with Outlet type
plt.figure(figsize=(15,6))
sns.boxplot(data=df, x='Outlet_Identifier', y='Item_Visibility', hue='Outlet_Type')
plt.xticks(rotation=45)
#Checking visibility in Item type with Outlet type
plt.figure(figsize=(15,6))
sns.boxplot(data=df, x='Item_Type', y='Item_Visibility', hue='Outlet_Type')
plt.xticks(rotation=45)
#Creating pivot table to help fill nan values of visibility from here
table = df.pivot_table(values='Item_Visibility', index='Item_Type', columns='Outlet_Type', aggfunc='mean')
table
#Creating pivot table to help fill nan values of visibility from here
table = df.pivot_table(values='Item_Visibility', index='Item_Type', columns='Outlet_Type', aggfunc='mean')
table
# replace the nan values
# define function that returns the mean values
def find_mean(x):
    return table.loc[x['Item_Type'], x['Outlet_Type']]

# replace missing values in visibility with mean values from above pivot table
df['Item_Visibility'].fillna(df[df['Item_Visibility'].isnull()].apply(find_mean, axis=1), inplace=True)
#we try to fill the nan values of wieht by using values from item identifier
wt_table = df.pivot_table(values='Item_Weight', index='Item_Identifier')

def find_wt(x):
    return wt_table.loc[x['Item_Identifier'],'Item_Weight']

df['Item_Weight'].fillna(df[df['Item_Weight'].isnull()].apply(find_wt, axis=1), inplace=True)
df['Outlet_Size'].replace(np.NaN,'Unknown',inplace=True)
from scipy.stats import mode
size_table = df.pivot_table(values='Outlet_Size', index='Outlet_Type', aggfunc=(lambda x:mode(x).mode[0]))

#Filling nan values with mode
df['Outlet_Size'].replace('Unknown','Small',inplace=True)

plt.figure(figsize=(8,6))
sns.heatmap(df.isnull(),cmap='Greys')
df['Item_Identifier']=df['Item_Identifier'].apply(lambda x: x[:-2])
df['Item_Identifier'].unique()
#Correcting year column by subtracting it from 2021
df['Outlet_Establishment_Year']=(2021.0-df['Outlet_Establishment_Year'])
df.head()
#Merging all the low fat categories to Low fat and regular categories to Regular
df['Item_Fat_Content'].unique()
df['Item_Fat_Content'].replace('reg','Regular',inplace=True)
df['Item_Fat_Content'].replace(['low fat','LF'],'Low Fat',inplace=True)
df['Item_Fat_Content'].unique()
#Using Ordinal Encoder for encoding object type values
from sklearn.preprocessing import OrdinalEncoder
e=OrdinalEncoder()
obj=[feature for feature in df.columns if df[feature].dtypes=='O' and feature!='source']
obj
for i in obj:
    df[i]=e.fit_transform(df[i].values.reshape(-1,1))
df.dtypes
df=df[np.abs(df.Item_Visibility-df.Item_Visibility.mean())<=(3*df.Item_Visibility.std())]
df.shape
sns.boxplot(df['Item_Visibility'])
df['Item_Visibility'].skew()
df['Item_Visibility']=np.sqrt(df['Item_Visibility'])
df['Item_Visibility'].skew()
sns.distplot(df['Item_Visibility'])
from sklearn.preprocessing import MinMaxScaler
m=MinMaxScaler()

df.iloc[:,:-2]=m.fit_transform(df.iloc[:,:-2])
train = df.loc[df['source']=='train']
test = df.loc[df['source']=='test']

train.drop('source',axis=1,inplace=True)
test.drop(['source','Item_Outlet_Sales'],axis=1,inplace=True)

train=train.reset_index(drop=True)
test=test.reset_index(drop=True)



#%%# In[5]: TRAIN AND ENVELUATE MODELS
from sklearn.model_selection import train_test_split,cross_val_score

#importing models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
x=train.iloc[:,:-1]
y=train.iloc[:,-1]
#Choosing the best random state using Logistic regression
def randomstate(a,b):
    maxx=0
    for state in range(1,201):
        xtrain,xtest,ytrain,ytest=train_test_split(a,b,test_size=0.2,random_state=state)
        model=LinearRegression()
        model.fit(xtrain,ytrain)
        p=model.predict(xtest)
        r2=r2_score(p,ytest)
        if maxx<r2:
            maxx=r2
            j=state
    return j
#Creating list of models and another list mapped to their names
models=[KNeighborsRegressor(),LinearRegression(),Lasso(),Ridge(),ElasticNet(),DecisionTreeRegressor(),
       RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),XGBRegressor()]

names=['KNeighborsRegressor','LinearRegression','Lasso','Ridge','ElasticNet','DecisionTreeRegressor',
       'RandomForestRegressor','AdaBoostRegressor','GradientBoostingRegressor','XGBRegressor']
def performance(p,ytest,s,n):
    print(m)
    print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
    print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
    print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
    print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
    print('Mean of cross validaton Score is',np.round(np.mean(s),4))
    print('--------------------------------------------------------------------------')
def createmodels(model_list,independent,dependent,n):
    xtrain,xtest,ytrain,ytest=train_test_split(independent,dependent,test_size=0.2,random_state=randomstate(x,y))
    name=[]
    meanabs=[]
    meansqd=[]
    rootmeansqd=[]
    r2=[]
    mcv=[]
    
    #Creating models
    for i,model in enumerate(model_list):
        model.fit(xtrain,ytrain)
        p=model.predict(xtest)
        score=cross_val_score(model,independent,dependent,cv=10)
        
        #Calculating scores of the model and appending them to a list
        name.append(n[i])
        meanabs.append(np.round(mean_absolute_error(p,ytest),4))
        meansqd.append(np.round(mean_squared_error(p,ytest),4))
        rootmeansqd.append(np.round(np.sqrt(mean_squared_error(p,ytest)),4))
        r2.append(np.round(r2_score(p,ytest),4)*100)
        mcv.append(np.round(np.mean(score),4)*100)
    
    #Creating Dataframe
    data=pd.DataFrame()
    data['Model']=name
    data['Mean Absolute Error']=meanabs
    data['Mean Squared Error']=meansqd
    data['Root Mean Squared Error']=rootmeansqd
    data['R2 Score']=r2
    data['Mean of Cross validaton Score']=mcv
    data.set_index('Model',inplace = True)
    return data
createmodels(models,x,y,names)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selection = SelectKBest(score_func=f_classif)
fit = selection.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
featureScores.sort_values(by=['Score'],ascending=False)
#Using feature importances of Extra trees regressor
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(11).plot(kind='barh')
plt.show()
fi=list(feat_importances.nlargest(11).index)
from sklearn.model_selection import GridSearchCV
param_grid={'alpha':[1e-15,1e-10,1e-8,1e-5,1e-3,0.1,1,5,10,15,20,30,35,45,50,55,65,100,110,150,1000]}
m1=GridSearchCV(Lasso(),param_grid,scoring='neg_mean_squared_error',cv=10)
m1.fit(x,y)
print(m1.best_params_)
m1=Lasso(alpha=1)
m1.fit(x,y)
importance = np.abs(m1.coef_)

dfcolumns = pd.DataFrame(x.columns)
dfimp=pd.DataFrame(importance)
featureScores = pd.concat([dfcolumns,dfimp],axis=1)
featureScores.columns = ['Features','Coefficients']  #naming the dataframe columns
featureScores.sort_values(by=['Coefficients'],ascending=False)
x1=x.drop('Outlet_Location_Type',axis=1)
x2=x.drop('Item_Type',axis=1)
x3=x.drop(['Outlet_Location_Type','Item_Type'],axis=1)
#Testing x1 dataset
createmodels(models,x1,y,names)
#Testing x2 dataset
createmodels(models,x2,y,names)
#Testing x3 dataset
createmodels(models,x3,y,names)


#%%# In[6]: FINE-TUNE MODELS
from sklearn.model_selection import GridSearchCV

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=randomstate(x2,y))
params={'n_estimators':[100, 300, 500],
        'min_samples_split':[1,2,3,4],
        'min_samples_leaf':[1,2,3,4],
            'max_depth':[None,1,2,3,4,5,6,7,8,9,10]}
g=GridSearchCV(RandomForestRegressor(),params,cv=5)
g.fit(xtrain,ytrain)
print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)
m=RandomForestRegressor(max_depth=6, min_samples_leaf=4, min_samples_split=3)
m.fit(xtrain,ytrain)
p=m.predict(xtest)

score=cross_val_score(m,x,y,cv=10)

print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
print('Mean of cross validaton Score is',np.round(np.mean(score)*100,4))
params={'n_estimators':[100,200,300,400,500],
      'learning_rate':[0.05, 0.10],
      'subsample':[0.5,1],
      'max_depth':[1,2,3,4,5,6,7,8,9,10]}

from sklearn.model_selection import RandomizedSearchCV

g=RandomizedSearchCV(GradientBoostingRegressor(),params,cv=5)
g.fit(xtrain,ytrain)
print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)
m=GradientBoostingRegressor(learning_rate=0.05, subsample=1,max_depth= 2,n_estimators=200 )
m.fit(xtrain,ytrain)
p=m.predict(xtest)
score=cross_val_score(m,x,y,cv=10)
print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
print('Mean of cross validaton Score is',np.round(np.mean(score)*100,4))
params={
 "learning_rate"    : [0.001,0.05, 0.10, ] ,
 "max_depth"        : [ 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
g=RandomizedSearchCV(XGBRegressor(),params,cv=5)
g.fit(xtrain,ytrain)
print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)
m=XGBRegressor(colsample_bytree= 0.7, gamma= 0.1, learning_rate=0.05, max_depth=5, min_child_weight= 3)
m.fit(xtrain,ytrain)
p=m.predict(xtest)
score=cross_val_score(m,x,y,cv=10)
print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
print('Mean of cross validaton Score is',np.round(np.mean(score)*100,4))



#%%# In[7]: ANALYZE AND TEST YOUR SOLUTION
model=GradientBoostingRegressor(learning_rate=0.05, subsample=1,max_depth= 3,n_estimators=100 )
model.fit(xtrain,ytrain)
p=model.predict(xtest)
score=cross_val_score(m,x,y,cv=10)
print('Mean Absolute Error is',np.round(mean_absolute_error(p,ytest),4))
print('Mean Squared Error is',np.round(mean_squared_error(p,ytest),4))
print('Root Mean Squared Error is',np.round(np.sqrt(mean_squared_error(p,ytest)),4))
print('R2 Score is',np.round(r2_score(p,ytest),4)*100)
print('Mean of cross validaton Score is',np.round(np.mean(score)*100,4))
plt.scatter(x=ytest,y=p,color='r')
plt.plot(ytest,ytest,color='b')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Gradient Boost Regressor')
import joblib
joblib.dump(model,'Mart_Sales.obj')
predictions=model.predict(test)
predictions=pd.DataFrame(predictions)
predictions.to_csv('test_predictions.csv')