#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#%%
df= pd.read_csv(r'C:/Users/Abhijeet Sant/Downloads/XYZCorp_LendingData.txt', sep='\t',low_memory=False)

#%%
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

#%%
half_count=len(df)/2
df=df.dropna(thresh=half_count,axis=1)
#%%
df.describe(include='all')
#%%
# =============================================================================
# loan_amnt= df['loan_amnt']
# plt.hist(loan_amnt)
# plt.plot()
# #%%
# 'annual_inc'
# #%%
# loan_amnt= df['annual_inc']
# plt.hist(loan_amnt)
# plt.plot()
# #%%
# sns.distplot(loan_amnt)
# plt.plot()
# #%%
# sns.boxplot(x=df['annual_inc'])
# #%%
# annual_inc=df['annual_inc']
# sns.distplot(annual_inc)
# plt.plot()
# #%%
# plt.boxplot(df)
# plt.plot()
# =============================================================================
#%%
df.columns
df.shape
#%%
df.head()
#%%
df.columns
#%%
drop_list=['id', 'member_id','funded_amnt', 'funded_amnt_inv','int_rate','sub_grade', 'emp_title','total_rec_prncp', 'total_rec_int',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'last_pymnt_d', 'last_pymnt_amnt','zip_code','out_prncp', 'out_prncp_inv','total_pymnt',
       'total_pymnt_inv']
#%%
df=df.drop(drop_list,axis=1)
# =============================================================================
# #%%
# drop_cols=['zip_code','out_prncp', 'out_prncp_inv','total_pymnt',
#        'total_pymnt_inv',]
# #%%
# df=df.drop(drop_cols,axis=1)
# #%%
# drop_cols=['total_rec_prncp', 'total_rec_int',
#        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
#        'last_pymnt_d', 'last_pymnt_amnt']
# #%%
# df=df.drop(drop_cols,axis=1)
# =============================================================================
#%%
df.drop(['next_pymnt_d',],axis=1,inplace=True)
#%%
df.drop(['last_credit_pull_d',],axis=1,inplace=True)
#%%
df.drop(['earliest_cr_line',],axis=1,inplace=True)
#%%
df.columns
#%%
df.shape
#%%

#%%
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='default_ind',data=df,ax=axs[0])
axs[0].set_title("Frequency of each Loan Status")
df.default_ind.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Loan status")
plt.show()
#%%
df = df.loc[:,df.apply(pd.Series.nunique) != 1]
#%%
for col in df.columns:
    if (len(df[col].unique()) < 4):
        print(df[col].value_counts())
        print()
df.shape
#%%
df.isnull().sum()
#%%
ax = sns.distplot(df['loan_amnt'], bins =10, kde=False, color="b", axlabel='loan amount')
#%%
group = df.groupby('grade').agg([np.median])
loanamount = group['loan_amnt'].reset_index()

ax = sns.barplot(y = "median", x = 'grade', data=loanamount)
ax.set(xlabel = 'loan grade', ylabel = 'median loan amount', title = 'median loan amount, by loan grade')
#%%
df.isnull().sum()
#%%
#df.drop(['next_pymnt_d',],axis=1,inplace=True)
#%%
df.shape
df.isnull().sum()
#%%
df.describe(include='all')
#%%
df.shape
#%%
df.select_dtypes(include=['float']).dtypes
#%%
list(df.columns)
#%%

#%%

#%%
## combine these different collections into a list    
data_to_plot = ['annual_inc','delinq_2yrs','inq_last_6mths',
 'open_acc',
 'pub_rec',
 'revol_bal',
 'revol_util',
 'total_acc','collections_12_mths_ex_med','acc_now_delinq','tot_cur_bal','total_rev_hi_lim']
#%%
# Create a figure instance
fig = plt.figure(1, figsize=(9,6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)

# Save the figure
#fig.savefig('fig1.png', bbox_inches='tight')
#%%
df.isnull().sum()
# =============================================================================
# #%%
# df.shape
# #
# class sklearn.preprocessing.RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
# #
# std = RobustScaler()
# std.fit(df)
# df = std.transform(df)
# =============================================================================
#%%
df['title'] = np.where(df['title'].isnull(), 'title not given', df['title'])
#%%
df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())
#%%
df['collections_12_mths_ex_med'] = np.where(df['collections_12_mths_ex_med'].isnull(), 0, df['collections_12_mths_ex_med'])
#%%
df['tot_coll_amt'] = df['tot_coll_amt'].fillna(df['tot_coll_amt'].median())
df['tot_cur_bal'] = df['tot_cur_bal'].fillna(df['tot_cur_bal'].median())
df['total_rev_hi_lim'] = df['total_rev_hi_lim'].fillna(df['total_rev_hi_lim'].median())
#%%
#df.drop(['last_credit_pull_d',],axis=1,inplace=True)
#%%
#df.drop(['earliest_cr_line',],axis=1,inplace=True)
#%%
df.shape
#%%
df.isnull().sum()
#%%
##Boxplot:
df_1 = pd.DataFrame(data = np.random.random(size=(12,12)), columns = ['annual_inc','delinq_2yrs','inq_last_6mths',
 'open_acc',
 'pub_rec',
 'revol_bal',
 'revol_util',
 'total_acc','collections_12_mths_ex_med','acc_now_delinq','tot_cur_bal','total_rev_hi_lim'])
df_1.boxplot()
#%%
import seaborn as sns

#sns.boxplot(df["installment"])
sns.boxplot( df["annual_inc"])
#sns.boxplot( x=df["delinq_2yrs"])
#sns.boxplot( x=df["inq_last_6mths"])
#sns.boxplot(df["open_acc"])
#sns.boxplot( x=df["pub_rec"])
#sns.boxplot( x=df["total_rev_hi_lim"])
#sns.boxplot( x=df["tot_cur_bal"])
#sns.boxplot( x=df["acc_now_delinq"])
#sns.boxplot( x=df["collections_12_mths_ex_med"])
#sns.boxplot( x=df["total_acc"])
#sns.boxplot( x=df["revol_bal"])
#sns.boxplot( x=df["revol_util"])

#sns.boxplot(data=df.iloc[:,0:6])
#%%
df = df[df.revol_util <= 193]
#%%
df = df[df.pub_rec <=63]
#%%
df =df[df.annual_inc <= 2039784]
#%%
#df['revol_util'].value_counts
#%%
df=df[df.acc_now_delinq <= 6]
#%%
df.columns
#%%
plt.boxplot(['installment']).mean()

#%%
# extract numbers from emp_length and fill missing values with the median
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)
df['emp_length'] = df['emp_length'].fillna(df.emp_length.median())

#%%
df.shape
#%%
df.columns
# =============================================================================
# #%%
# col_dates = df.dtypes[df.dtypes == 'datetime64[ns]'].index
# for d in col_dates:
#     df[d] = df[d].dt.to_period('M')
# =============================================================================
#%%
colabel=['term','grade','emp_length','home_ownership','verification_status','purpose','title','addr_state','initial_list_status','pymnt_plan','application_type']
#%%
colabel
#%%
from sklearn import preprocessing

le={}

for x in colabel:
    le[x]=preprocessing.LabelEncoder()
    
for x in colabel:
    df[x]=le[x].fit_transform(df.__getattr__(x))

#%%

#%%
split=['June-2015','July-2015','Aug-2015','Sept-2015','Oct-2015','Nov-2015','Dec-2015']
#%%
test_data=df.loc[df.issue_d.isin(split)]
train_data=df.loc[-df.issue_d.isin(split)]
#%%
print(test_data.shape)
train_data.shape
#%%
test_data.drop(['issue_d'],axis=1,inplace=True)
train_data.drop(['issue_d'],axis=1,inplace=True)
#%%
print(train_data.shape)
test_data.shape
#%%

#%%
df.describe()
#%%
X_Train=train_data.values[:,:-1]
Y_Train=train_data.values[:,-1]
X_Test=test_data.values[:,:-1]
Y_Test=test_data.values[:,-1]
#%%
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X_Train)
X_Train=scaler.transform(X_Train)

scaler.fit(X_Test)
X_Test=scaler.transform(X_Test)
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,\
classification_report
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
def run_models(X_Train, Y_Train, X_Test, Y_Test, model_type = 'Non-balanced'):
    
    clfs = {'LogisticRegression' : LogisticRegression(),'DecisionTreeClassifier': DecisionTreeClassifier(),'RandomForestClassifier': RandomForestClassifier(n_estimators=10) }
    cols = ['model','roc_auc_score', 'precision_score', 'recall_score','f1_score','Accuracy','Confusion_Matrix']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()

    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        clf.fit(X_Train, Y_Train)

        y_pred = clf.predict(X_Test)
        y_score = clf.predict_proba(X_Test)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,
                         'roc_auc_score' : metrics.roc_auc_score(Y_Test, y_score),
                         'precision_score': metrics.precision_score(Y_Test, y_pred),
                         'recall_score': metrics.recall_score(Y_Test, y_pred),
                         'f1_score': metrics.f1_score(Y_Test, y_pred),
                         'confusion_matrix' :confusion_matrix(Y_Test,y_pred),
                         'Accuracy' : accuracy_score(Y_Test,y_pred)})

        models_report = models_report.append(tmp, ignore_index = True)
        conf_matrix[clf_name] = pd.crosstab(Y_Test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
        fpr, tpr, thresholds = metrics.roc_curve(Y_Test, y_score, drop_intermediate = False, pos_label = 1)

        plt.figure(1, figsize=(6,6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve - {}'.format(model_type))
        plt.plot(fpr, tpr, label = clf_name )
        plt.legend(loc=2, prop={'size':11})
    plt.plot([0,1],[0,1], color = 'black')
    
    return models_report, conf_matrix
#%%
#%%

models_report, conf_matrix = run_models(X_Train, Y_Train, X_Test, Y_Test)
#%%
models_report
#%%
classifier=(LogisticRegression())
from sklearn import cross_validation

kfold_cv=cross_validation.KFold(n=len(X_Train),n_folds=10)
print(kfold_cv)

kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_Train,y=Y_Train,scoring="accuracy",cv=kfold_cv)
print(kfold_cv_result)
print(kfold_cv_result.mean())
#%%
classifier=(DecisionTreeClassifier())
from sklearn import cross_validation

kfold_cv=cross_validation.KFold(n=len(X_Train),n_folds=10)
print(kfold_cv)

kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_Train,y=Y_Train,scoring="accuracy",cv=kfold_cv)
print(kfold_cv_result)
print(kfold_cv_result.mean())
#%%
# =============================================================================
# classifier=(RandomForestClassifier(n_estimators=10))
# from sklearn import cross_validation
# 
# kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
# print(kfold_cv)
# 
# kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,y=Y_train,scoring="accuracy",cv=kfold_cv)
# print(kfold_cv_result)
# print(kfold_cv_result.mean())
# #%%
# classifier=(DecisionTreeClassifier())
# from sklearn import cross_validation
# 
# kfold_cv=cross_validation.KFold(n=len(X_Train),n_folds=10)
# print(kfold_cv)
# 
# kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_Train,y=Y_Train,scoring="accuracy",cv=kfold_cv)
# print(kfold_cv_result)
# print(kfold_cv_result.mean())
# #%%
# from sklearn.tree import DecisionTreeClassifier
# model_DecisionTree=DecisionTreeClassifier()
# model_DecisionTree.fit(X_Train,Y_Train)
# 
# Y_pred=model_DecisionTree.predict(X_Test)
# #%%
# from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
# print(confusion_matrix(Y_Test,Y_pred))
# print(accuracy_score(Y_Test,Y_pred))
# print(classification_report(Y_Test,Y_pred))
# =============================================================================
#%%
classifier=(RandomForestClassifier(n_estimators=10))
from sklearn import cross_validation

kfold_cv=cross_validation.KFold(n=len(X_Train),n_folds=10)
print(kfold_cv)

kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_Train,y=Y_Train,scoring="accuracy",cv=kfold_cv)
print(kfold_cv_result)
print(kfold_cv_result.mean())
#%%
# =============================================================================
# from sklearn.ensemble import RandomForestClassifier
# 
# model_RandomForest=RandomForestClassifier(501)
# 
# #fit the model on the data and predict the values
# model_RandomForest.fit(X_Train,Y_Train)
# 
# Y_pred=model_RandomForest.predict(X_Test)
# #%%
# from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
# print(confusion_matrix(Y_Test,Y_pred))
# print(accuracy_score(Y_Test,Y_pred))
# print(classification_report(Y_Test,Y_pred))
# 
# =============================================================================
#%%
# =============================================================================
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, accuracy_score,\
# classification_report
# from sklearn.model_selection import cross_val_score
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# def run_models(X_Train, Y_Train, X_Test, Y_Test, model_type = 'Non-balanced'):
#     
#     clfs = {'LogisticRegression' : LogisticRegression(),'RandomForestClassifier': RandomForestClassifier(n_estimators=10),'DecisionTreeClassifier': DecisionTreeClassifier() }
#     cols = ['model','roc_auc_score', 'precision_score', 'recall_score','f1_score','Accuracy','Confusion_Matrix']
# 
#     models_report = pd.DataFrame(columns = cols)
#     conf_matrix = dict()
# 
#     for clf, clf_name in zip(clfs.values(), clfs.keys()):
# 
#         clf.fit(X_Train, Y_Train)
# 
#         y_pred = clf.predict(X_Test)
#         y_score = clf.predict_proba(X_Test)[:,1]
# 
#         print('computing {} - {} '.format(clf_name, model_type))
# 
#         tmp = pd.Series({'model_type': model_type,
#                          'model': clf_name,
#                          'roc_auc_score' : metrics.roc_auc_score(Y_Test, y_score),
#                          'precision_score': metrics.precision_score(Y_Test, y_pred),
#                          'recall_score': metrics.recall_score(Y_Test, y_pred),
#                          'f1_score': metrics.f1_score(Y_Test, y_pred),
#                          'confusion_matrix' :confusion_matrix(Y_Test,y_pred),
#                          'Accuracy' : accuracy_score(Y_Test,y_pred)})
# 
#         models_report = models_report.append(tmp, ignore_index = True)
#         conf_matrix[clf_name] = pd.crosstab(Y_Test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
#         fpr, tpr, thresholds = metrics.roc_curve(Y_Test, y_score, drop_intermediate = False, pos_label = 1)
# 
#         plt.figure(1, figsize=(6,6))
#         plt.xlabel('false positive rate')
#         plt.ylabel('true positive rate')
#         plt.title('ROC curve - {}'.format(model_type))
#         plt.plot(fpr, tpr, label = clf_name )
#         plt.legend(loc=2, prop={'size':11})
#     plt.plot([0,1],[0,1], color = 'black')
#     
#     return models_report, conf_matrix
##%%
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#%%

#models_report, conf_matrix = run_models(X_Train, Y_Train, X_Test, Y_Test)
#%%
#models_report
# =============================================================================
#%%

#%%
from sklearn.utils import resample
#%%
train_data.default_ind.value_counts()

#%%
df_majority=train_data[train_data.default_ind==0]
df_minority=train_data[train_data.default_ind==1]
#%%
df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=653188,
                                     random_state=10)
#%%
df_upsampled=pd.concat([df_majority,df_minority_upsampled])
#%%
df_upsampled.default_ind.value_counts()
#%%
df_upsampled.value_counts()
#%%
df_upsampled.describe(include='all')
#%%
df_upsampled.head(8)
#%%
new_train=df_upsampled.values[:,:-1]
new_test=df_upsampled.values[:,-1]
#%%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(new_train,new_test, test_size =0.3, random_state=10)
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,\
classification_report
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
def run_models(x_train, y_train, X_Test, Y_Test, model_type = 'Balanced'):
    
    clfs = {'LogisticRegression' : LogisticRegression(),'DecisionTreeClassifier': DecisionTreeClassifier(),'RandomForestClassifier': RandomForestClassifier(n_estimators=10) }
    cols = ['model','roc_auc_score', 'precision_score', 'recall_score','f1_score','Accuracy']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()

    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        clf.fit(x_train, y_train)

        y_pred = clf.predict(X_Test)
        y_score = clf.predict_proba(X_Test)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,
                         'roc_auc_score' : metrics.roc_auc_score(Y_Test, y_score),
                         'precision_score': metrics.precision_score(Y_Test, y_pred),
                         'recall_score': metrics.recall_score(Y_Test, y_pred),
                         'f1_score': metrics.f1_score(Y_Test, y_pred),
                         'confusion_matrix' :confusion_matrix(Y_Test,y_pred),
                         'Accuracy' : accuracy_score(Y_Test,y_pred)})

        models_report = models_report.append(tmp, ignore_index = True)
        conf_matrix[clf_name] = pd.crosstab(Y_Test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
        fpr, tpr, thresholds = metrics.roc_curve(Y_Test, y_score, drop_intermediate = False, pos_label = 1)

        plt.figure(1, figsize=(6,6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve - {}'.format(model_type))
        plt.plot(fpr, tpr, label = clf_name )
        plt.legend(loc=2, prop={'size':11})
    plt.plot([0,1],[0,1], color = 'black')
    
    return models_report, conf_matrix
#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#%%

models_report, conf_matrix = run_models(x_train, y_train, X_Test, Y_Test)
#%%
models_report
# =============================================================================
# #%%
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, accuracy_score,\
# classification_report
# from sklearn.model_selection import cross_val_score
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# 
# y_pred = clf.predict(X_Test)
# y_score = clf.predict_proba(X_Test)[:,1]
# #%%
# from sklearn.linear_model import LogisticRegression
# #Create a model
# 
# classifier=(LogisticRegression())
# #fitting training data to the model
# classifier.fit(x_train,y_train)
# 
# Y_pred=classifier.predict(X_Test)
# #%%
# (list(Y_pred)).value_counts()
# #%%
# df_new = pd.DataFrame(Y_pred,columns=['Y_predictions'])
# #%%
# df_new.Y_predictions.value_counts()
# 
# =============================================================================
#%%
from sklearn.linear_model import LogisticRegression
#Create a model

classifier=(LogisticRegression())
#fitting training data to the model
classifier.fit(x_train,y_train)

y_pred_prob = classifier.predict_proba(X_Test)
print(y_pred_prob)
#%%
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.47:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
#print(y_pred_class)
#
from sklearn.metrics import confusion_matrix, accuracy_score
cfm=confusion_matrix(Y_Test.tolist(),y_pred_class)
# .test contains array and class contains list. so to convert array to list we use .tolist
print(cfm)
print("Classification report:")
print(classification_report(Y_Test,y_pred_class))
accuracy_score = accuracy_score(Y_Test.tolist(),y_pred_class)
print("Accuracy of the model: ",accuracy_score)
#
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(Y_Test,y_pred_class)
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# =============================================================================
# #%%
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, accuracy_score,\
# classification_report
# from sklearn.model_selection import cross_val_score
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier
# def run_models(x_train, y_train, x_test, y_test, model_type = 'Balanced'):
#     
#     clfs = {'LogisticRegression' : LogisticRegression(),'RandomForestClassifier': RandomForestClassifier(n_estimators=10),'DecisionTreeClassifier': DecisionTreeClassifier() }
#     cols = ['model','roc_auc_score', 'precision_score', 'recall_score','f1_score','Accuracy']
# 
#     models_report = pd.DataFrame(columns = cols)
#     conf_matrix = dict()
# 
#     for clf, clf_name in zip(clfs.values(), clfs.keys()):
# 
#         clf.fit(x_train, y_train)
# 
#         y_pred_class = clf.predict(X_Test)
#         y_score = clf.predict_proba(X_Test)[:,1]
# 
#         print('computing {} - {} '.format(clf_name, model_type))
# 
#         tmp = pd.Series({'model_type': model_type,
#                          'model': clf_name,
#                          'roc_auc_score' : metrics.roc_auc_score(Y_Test, y_score),
#                          'precision_score': metrics.precision_score(Y_Test, y_pred_class),
#                          'recall_score': metrics.recall_score(Y_Test, y_pred_class),
#                          'f1_score': metrics.f1_score(Y_Test, y_pred_class),
#                          'confusion_matrix' :confusion_matrix(Y_Test,y_pred_class),
#                          'Accuracy' : accuracy_score(Y_Test,y_pred_class)})
# 
#         models_report = models_report.append(tmp, ignore_index = True)
#         conf_matrix[clf_name] = pd.crosstab(Y_Test, y_pred_class, rownames=['True'], colnames= ['Predicted'], margins=False)
#         fpr, tpr, thresholds = metrics.roc_curve(Y_Test, y_score, drop_intermediate = False, pos_label = 1)
# 
#         plt.figure(1, figsize=(6,6))
#         plt.xlabel('false positive rate')
#         plt.ylabel('true positive rate')
#         plt.title('ROC curve - {}'.format(model_type))
#         plt.plot(fpr, tpr, label = clf_name )
#         plt.legend(loc=2, prop={'size':11})
#     plt.plot([0,1],[0,1], color = 'black')
#     
#     return models_report, conf_matrix
# 
# #%%
# models_report, conf_matrix = run_models(x_train, y_train, X_Test, Y_Test)
# #%%
# models_report
# #%%
# #%%
# classifier=(LogisticRegression())
# from sklearn import cross_validation
# 
# kfold_cv=cross_validation.KFold(n=len(X_Train),n_folds=10)
# print(kfold_cv)
# 
# kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,y=y_train,scoring="accuracy",cv=kfold_cv)
# print(kfold_cv_result)
# print(kfold_cv_result.mean())
# 
# for train_value,test_value in kfold_cv:
# 
#     classifier.fit(x_train[train_value],y_train[train_value]).predict_proba(x_train[test_value])
# 
#  
# 
# Y_pred=classifier.predict(X_Test)
# #%%
# from sklearn import metrics
# fpr, tpr, thresholds = metrics.roc_curve(Y_Test,Y_pred)
# roc_auc = metrics.auc(fpr, tpr)
# print("Area under the ROC curve : %f" % roc_auc)
# 
# 
# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# #%%
# from sklearn.metrics import roc_auc_score
# auc= roc_auc_score(Y_Test,Y_pred)
# print('AUC:',auc)
# #%%
# from sklearn.metrics import confusion_matrix, accuracy_score,\
# classification_report
# from sklearn.metrics import roc_auc_score
# cfm=confusion_matrix(Y_Test,Y_pred)
# print(cfm)
# auc = roc_auc_score(Y_Test,Y_pred)
# print(auc)
# print("Classification report:")
# print(classification_report(Y_Test,Y_pred))
# 
# accuracy_score=accuracy_score(Y_Test,Y_pred)
# print("Accuracy of the model:",accuracy_score)
# #%%
# 
# #%%
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, accuracy_score,\
# classification_report
# from sklearn.model_selection import cross_val_score
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier
# def run_models(x_train, y_train, x_test, y_test, model_type = 'Balanced'):
#     
#     clfs = LogisticRegression()
#     cols = ['model','roc_auc_score', 'precision_score', 'recall_score','f1_score','Accuracy']
# 
#     models_report = pd.DataFrame(columns = cols)
#     conf_matrix = dict()
# 
#     for clf, clf_name in clfs:
# 
#         clf.fit(x_train, y_train)
# 
#         y_pred = clf.predict(X_Test)
#         y_score = clf.predict_proba(X_Test)[:,1]
# 
#         print('computing {} - {} '.format(clf_name, model_type))
# 
#         tmp = pd.Series({'model_type': model_type,
#                          'model': clf_name,
#                          'roc_auc_score' : metrics.roc_auc_score(Y_Test, y_score),
#                          'precision_score': metrics.precision_score(Y_Test, Y_pred),
#                          'recall_score': metrics.recall_score(Y_Test, Y_pred),
#                          'f1_score': metrics.f1_score(Y_Test, Y_pred),
#                          'confusion_matrix' :confusion_matrix(Y_Test,Y_pred),
#                          'Accuracy' : accuracy_score(Y_Test,Y_pred)})
# 
#         models_report = models_report.append(tmp, ignore_index = True)
#         conf_matrix[clf_name] = pd.crosstab(Y_Test, Y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
#         fpr, tpr, thresholds = metrics.roc_curve(Y_Test, y_score, drop_intermediate = False, pos_label = 1)
# 
#         plt.figure(1, figsize=(6,6))
#         plt.xlabel('false positive rate')
#         plt.ylabel('true positive rate')
#         plt.title('ROC curve - {}'.format(model_type))
#         plt.plot(fpr, tpr, label = clf_name )
#         plt.legend(loc=2, prop={'size':11})
#     plt.plot([0,1],[0,1], color = 'black')
#     
#     return models_report, conf_matrix
# 
# #%%
# models_report, conf_matrix = run_models(x_train, y_train, X_Test, Y_Test)
# #%%
# models_report
# 
# #%%
#  'roc_auc_score' : metrics.roc_auc_score(Y_Test, y_score)
#                          'precision_score': metrics.precision_score(Y_Test, y_pred)
#                          'recall_score': metrics.recall_score(Y_Test, y_pred)
#                          'f1_score': metrics.f1_score(Y_Test, y_pred)
#                          'confusion_matrix' :confusion_matrix(Y_Test,y_pred)
#                          'Accuracy' : accuracy_score(Y_Test,y_pred)
# #%%
# print(Y_pred_col)
# #%%
# Y_pred_col=list(Y_pred)
# #%%
# test_data_new= pd.read_csv(r'C:/Users/Abhijeet Sant/Downloads/XYZCorp_LendingData.txt', sep='\t',low_memory=False)
# #df=pd.read_csv(r'C:/Users/Abhijeet Sant/Downloads/XYZCorp_LendingData.txt',engine = 'python',sep='/t',)
# test_data_new["Y_predictions"]=Y_pred_col
# test_data_new.head()
# #%%
# classifier=(DecisionTreeClassifier())
# from sklearn import cross_validation
# 
# kfold_cv=cross_validation.KFold(n=len(X_Train),n_folds=10)
# print(kfold_cv)
# 
# kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,y=y_train,scoring="accuracy",cv=kfold_cv)
# print(kfold_cv_result)
# print(kfold_cv_result.mean())
# #%%
# #%%
# classifier=(RandomForestClassifier(n_estimators=10))
# from sklearn import cross_validation
# 
# kfold_cv=cross_validation.KFold(n=len(X_Train),n_folds=10)
# print(kfold_cv)
# 
# kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,y=y_train,scoring="accuracy",cv=kfold_cv)
# print(kfold_cv_result)
# print(kfold_cv_result.mean())
# =============================================================================
