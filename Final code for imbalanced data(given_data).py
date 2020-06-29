# Imbalanced Data set Algorithm Implementations
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing train and test data


# Importing train and test data

df_train=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Raw Data\\train_indessa.csv")
df_test=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Raw Data\\test_indessa.csv")

# Step 1 -  Data Cleaning and removing null values column by column

# 1. member_id will not have an impact on a person defaulting loan- But at the end we will have to submit data based on member_id- for that purpose I am just storing the member_id column of Test data in a seprate data frame called submissions


submission=df_test["member_id"]

df_train=df_train.drop("member_id",axis=1)
df_test=df_test.drop("member_id",axis=1)

# 2. Loan Amount- Do not have any null values and the variable has a direct impact on the Loan Default Status

df_train["loan_amnt"].isnull().sum()
df_test["loan_amnt"].isnull().sum()

# 3. Loan Amount Sanctioned by the Bank- Important Feature checking if there are any null values:
df_train["funded_amnt"].isnull().sum()
df_test["funded_amnt"].isnull().sum()

#4. Loan amount sanctioned by Investors- Important numerical feature checking if there are null values:

df_train["funded_amnt_inv"].isnull().sum()
df_test["funded_amnt_inv"].isnull().sum()

#5. Term - of the Loan in Months

df_train["term"].isnull().sum()
df_test["term"].isnull().sum()

# Transform the term column into numerical data
df_train["term"].replace(to_replace="months",value="",regex=True,inplace=True)
df_train["term"]=pd.to_numeric(df_train["term"])
type(df_train["term"][0])
df_train["term"][0]

df_test["term"].replace(to_replace="months",value="",regex=True,inplace=True)
df_test["term"]=pd.to_numeric(df_test["term"])
type(df_test["term"][0])
df_test["term"][0]

# 6. Batch Enrolled- Batch number allocated to members- It do not have an impact on loan status so droping the same

df_train=df_train.drop("batch_enrolled",axis=1)
df_test=df_test.drop("batch_enrolled",axis=1)

# 7. Interest Rate- it has an impact on loan Status- There are no null values and the dattype is float

df_train["int_rate"].isnull().sum()
df_test["int_rate"].isnull().sum()

# 8. Grade & 9 Subgrade- Grade assigned by the bank-  We can combine both and use one column if we replace the alphabetic grade in subgrade by a numeric value
df_train["grade"].value_counts()
df_train["sub_grade"].value_counts()

df_train["sub_grade"].replace(to_replace="A",value="1",regex=True,inplace=True)
df_train["sub_grade"].replace(to_replace="B",value="2",regex=True,inplace=True)
df_train["sub_grade"].replace(to_replace="C",value="3",regex=True,inplace=True)
df_train["sub_grade"].replace(to_replace="D",value="4",regex=True,inplace=True)
df_train["sub_grade"].replace(to_replace="E",value="5",regex=True,inplace=True)
df_train["sub_grade"].replace(to_replace="F",value="6",regex=True,inplace=True)
df_train["sub_grade"].replace(to_replace="G",value="7",regex=True,inplace=True)

df_test["sub_grade"].replace(to_replace="A",value="1",regex=True,inplace=True)
df_test["sub_grade"].replace(to_replace="B",value="2",regex=True,inplace=True)
df_test["sub_grade"].replace(to_replace="C",value="3",regex=True,inplace=True)
df_test["sub_grade"].replace(to_replace="D",value="4",regex=True,inplace=True)
df_test["sub_grade"].replace(to_replace="E",value="5",regex=True,inplace=True)
df_test["sub_grade"].replace(to_replace="F",value="6",regex=True,inplace=True)
df_test["sub_grade"].replace(to_replace="G",value="7",regex=True,inplace=True)

df_train["sub_grade"]=pd.to_numeric(df_train["sub_grade"])
df_test["sub_grade"]=pd.to_numeric(df_test["sub_grade"])

# Removing the grade

df_train=df_train.drop("grade",axis=1)
df_test=df_test.drop("grade",axis=1)

#10- Employee Title- It do not have an impact on the loan default behaviour of custtomer

df_train=df_train.drop(["emp_title"],axis=1)
df_test=df_test.drop(["emp_title"],axis=1)

# 11. Employement Length- 

df_train["emp_length"].replace("n/a","0",regex=True,inplace=True)
df_train["emp_length"].replace("years"," ",regex=True,inplace=True)
df_train["emp_length"].replace("< 1 year","0",regex=True,inplace=True)
df_train["emp_length"].replace("\+ years"," ",regex=True,inplace=True)
df_train["emp_length"].replace("year"," ",regex=True,inplace=True)


df_test["emp_length"].replace("n/a","0",regex=True,inplace=True)
df_test["emp_length"].replace("years"," ",regex=True,inplace=True)
df_test["emp_length"].replace("< 1 year","0",regex=True,inplace=True)
df_test["emp_length"].replace("\+ years"," ",regex=True,inplace=True)
df_test["emp_length"].replace("year"," ",regex=True,inplace=True)

df_train["emp_length"]=pd.to_numeric(df_train["emp_length"],errors="coerce")
df_test["emp_length"]=pd.to_numeric(df_test["emp_length"],errors="coerce")

# Removing the null values from emp_length with the median value

df_train["emp_length"].fillna(df_train["emp_length"].median(),inplace=True)
df_test["emp_length"].fillna(df_test["emp_length"].median(),inplace=True)


# 12. Home Ownership- Status of Home Ownership- Do not have that relavence



# Deleting the Homeownership Column

df_train=df_train.drop(["home_ownership"],axis=1)

df_test=df_test.drop(["home_ownership"],axis=1)



# 13. Annual Income- It has a relation with Loan Status- For null values putting the median


df_train["annual_inc"].fillna(df_train["annual_inc"].median(),inplace=True)
df_test["annual_inc"].fillna(df_test["annual_inc"].median(),inplace=True)

# 14. verification_status- Do not have that relevance

df_train=df_train.drop(["verification_status"],axis=1)

df_test=df_test.drop(["verification_status"],axis=1)


#15 pymnt plan- Not considering this for the time being because the data distribution is higly imbalanced in this case

df_train["pymnt_plan"].value_counts()

# Droping payement plan

df_train=df_train.drop("pymnt_plan",axis=1)
df_test=df_test.drop("pymnt_plan",axis=1)

#16 Desc- Loan reason provided by member- Not relavent droping the same

df_train=df_train.drop("desc",axis=1)
df_test=df_test.drop("desc",axis=1)

#17 & 18 Purppose & Title- Not relavent removing the same

df_train=df_train.drop(["purpose","title"],axis=1)
df_test=df_test.drop(["purpose","title"],axis=1)

# 19 & 20 Zip code and addr_ state
df_train=df_train.drop(["zip_code","addr_state"],axis=1)
df_test=df_test.drop(["zip_code","addr_state"],axis=1)

#21- DTI- ratio of member's total monthly debt repayment excluding mortgage divided by self reported monthly income

df_train["dti"].isnull().sum()
df_test["dti"].isnull().sum()

#22 delinq_2yrs- Condidering that for the timebeing
df_train["delinq_2yrs"].isnull().sum()
df_train["delinq_2yrs"].value_counts()

# replacing the null values with median
df_train["delinq_2yrs"].fillna(df_train["delinq_2yrs"].median(),inplace=True)
df_test["delinq_2yrs"].fillna(df_test["delinq_2yrs"].median(),inplace=True)

# 23 inq_last_6mnths- Not relavent with regards to Loan Default status

df_train=df_train.drop(["inq_last_6mths"],axis=1)

df_test=df_test.drop(["inq_last_6mths"],axis=1)

#24 mths_since_last_delinq - Replacing null values with median

df_train["mths_since_last_delinq"].fillna(df_train["mths_since_last_delinq"].median(),inplace=True)

df_test["mths_since_last_delinq"].fillna(df_test["mths_since_last_delinq"].median(),inplace=True)

# 25 mths_since_last_record- replacing  the null values with  median

df_train["mths_since_last_record"].fillna(df_train["mths_since_last_record"].median(),inplace=True)

df_test["mths_since_last_record"].fillna(df_test["mths_since_last_record"].median(),inplace=True)

#26 open_acc-Relavent removing the null values with meadian value

df_train["open_acc"].fillna(df_train["open_acc"].median(),inplace=True)

df_test["open_acc"].fillna(df_test["open_acc"].median(),inplace=True)

# 27 pub_rec- not that relevant

df_train=df_train.drop("pub_rec",axis=1)
df_test=df_test.drop("pub_rec",axis=1)

# 28 revol_bal - removing null values with median

df_train["revol_bal"].fillna(df_train["revol_bal"].median(),inplace=True)

df_test["revol_bal"].fillna(df_test["revol_bal"].median(),inplace=True)

# 29 revol_util- replacing the null values with median

df_train["revol_util"].fillna(df_train["revol_util"].median(),inplace=True)

df_test["revol_util"].fillna(df_test["revol_util"].median(),inplace=True)

#30-total_acc- removing the null values with median.

df_train["total_acc"].fillna(df_train["total_acc"].median(),inplace=True)

df_test["total_acc"].fillna(df_test["total_acc"].median(),inplace=True)

#31- initial_list_status

df_train=df_train.drop(["initial_list_status"],axis=1)
df_test=df_test.drop(["initial_list_status"],axis=1)

# 32- total_rec_int- removing the nulll values with median

df_train["total_rec_int"].fillna(df_train["total_rec_int"].median(),inplace=True)

df_test["total_rec_int"].fillna(df_test["total_rec_int"].median(),inplace=True)

# 33- total_rec_late_fee- Important parameter

#34 recoveries & 35- collection_recovery_fee has to be considered

#36-collections_12_mths_ex_med- removing the null value with zeros

df_train["collections_12_mths_ex_med"].fillna(0,inplace=True)

df_test["collections_12_mths_ex_med"].fillna(0,inplace=True)

#37- mths_since_last_major_derog- replace null with median()

df_train["mths_since_last_major_derog"].fillna(df_train["mths_since_last_major_derog"].median(),inplace=True)

df_test["mths_since_last_major_derog"].fillna(df_test["mths_since_last_major_derog"].median(),inplace=True)

#38- & 39-'application_type', 'verification_status_joint'- dropping because of irrelavance

df_train=df_train.drop(['application_type', 'verification_status_joint'],axis=1)
df_test=df_test.drop(['application_type', 'verification_status_joint'],axis=1)

#40-last_week_pay- not related to loan default status

df_train=df_train.drop(["last_week_pay"],axis=1)
df_test=df_test.drop(["last_week_pay"],axis=1)

#41-acc_now_delinq- replacing with median

df_train["acc_now_delinq"].fillna(df_train["acc_now_delinq"].median(),inplace=True)

df_test["acc_now_delinq"].fillna(df_test["acc_now_delinq"].median(),inplace=True)


#42-tot_coll_amt-replacing with median

df_train["tot_coll_amt"].fillna(df_train["tot_coll_amt"].median(),inplace=True)

df_test["tot_coll_amt"].fillna(df_test["tot_coll_amt"].median(),inplace=True)


#43-tot_cur_bal-replacing with median

df_train["tot_cur_bal"].fillna(df_train["tot_cur_bal"].median(),inplace=True)

df_test["tot_cur_bal"].fillna(df_test["tot_cur_bal"].median(),inplace=True)

#44total_rev_hi_lim-replacing with median

df_train["total_rev_hi_lim"].fillna(df_train["total_rev_hi_lim"].median(),inplace=True)

df_test["total_rev_hi_lim"].fillna(df_test["total_rev_hi_lim"].median(),inplace=True)

# Without removing any records null values have been replaced and relavent features have been identified

df_train.isnull().sum() # All null values have been removed

df_train.columns ### Please find below all features selected for the analysis
"""'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
       'sub_grade', 'emp_length', 'annual_inc', 'dti', 'delinq_2yrs',
       'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
       'revol_bal', 'revol_util', 'total_acc', 'total_rec_int',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'collections_12_mths_ex_med', 'mths_since_last_major_derog',
       'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
       'loan_status']"""

df_test.columns  ### Please find below all features selected for the analysis
"""['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
       'sub_grade', 'emp_length', 'annual_inc', 'dti', 'delinq_2yrs',
       'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
       'revol_bal', 'revol_util', 'total_acc', 'total_rec_int',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'collections_12_mths_ex_med', 'mths_since_last_major_derog',
       'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']"""

# Saving the file into train and test

df_train.to_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\train.csv")
df_test.to_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\test.csv")




# Step 4. Algorithm Implementation- 7 Algorithms have been Implemented and their accuracy have been mentioned

# a. Logistic regression

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\train.csv")
test=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\test.csv")

X=train.drop("loan_status",axis=1)
Y=train["loan_status"]

# Standardizing the data(Not required for Logistic regression but still doing it to reduce execution time)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)

#Implementation

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
model=lr.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import roc_curve,accuracy_score,auc

logistic_fpr,logistic_tpr,threshold=roc_curve(Y_test,Y_pred)

print(auc(logistic_fpr,logistic_tpr)) # 57.90

print (accuracy_score(Y_test,Y_pred)) # 77.87



#b. K-Nearest Neighbour

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\train.csv")
test=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\test.csv")


# Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)

#Implementation

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=1)
model=KNN.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test,Y_pred))

# Finding the optimum value of orginal Neighbors- Using Cross validation technique



errorrate=[]

for i in (1,300):
    KNN=KNeighborsClassifier(n_neighbors=i)
    model=KNN.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    errorrate.append(np.mean(Y_pred != Y_test))

# Finding the accuracy with optimum K value
    
KNN=KNeighborsClassifier(n_neighbors=241)
model=KNN.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,auc,roc_curve
knn_fpr,knn_tpr,threshold=roc_curve(Y_test,Y_pred)
print(auc(knn_fpr,knn_tpr))- # 94.10
print(accuracy_score(Y_test,Y_pred)) #94.10


#c.Naive Bayes

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\train.csv")
test=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\test.csv")


X=train.drop("loan_status",axis=1)
Y=train["loan_status"]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
model=nb.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score, auc,roc_curve
nb_fpr,nb_tpr,threshold=roc_curve(Y_test,Y_pred)
print(auc(nb_fpr,nb_tpr))       # 53%
print(accuracy_score(Y_test,Y_pred)) # 51%


#d.Support Vector Machines

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\train.csv")
test=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\test.csv")



X=train.drop("loan_status",axis=1)
Y=train["loan_status"]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)

from sklearn.svm import SVC
SVM=SVC()
model=SVM.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,auc,roc_curve
sv_fpr,sv_tpr,threshold=roc_curve(Y_test,Y_pred)
print(auc(sv_fpr,sv_tpr)) # 58.50
print(accuracy_score(Y_test,Y_pred)) # 75.3

#e. Descision Tree

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\train.csv")
test=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\test.csv")




X=train.drop("loan_status",axis=1)
Y=train["loan_status"]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)


from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
model=dt.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,auc,roc_curve
dt_fpr,dt_tpr,threshold=roc_curve(Y_test,Y_pred)
print(auc(dt_fpr,dt_tpr))  # 74.99%
print(accuracy_score(Y_test,Y_pred)) #81.62%

#f.Ensembling Techniques-Bagging- Random Forest

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\train.csv")
test=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\test.csv")

X=train.drop("loan_status",axis=1)
Y=train["loan_status"]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model=rf.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,auc,roc_curve

rf_fpr,rf_tpr,threshold=roc_curve(Y_test,Y_pred)
print(auc(rf_fpr,rf_tpr)) #67.43 % 
print(accuracy_score(Y_test,Y_pred)) #83.98 % 

#g.XgBoost- BOOSTING-XgBoost

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\train.csv")
test=pd.read_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\Cleaned data with imbalancing\\test.csv")


X=train.drop("loan_status",axis=1)
Y=train["loan_status"]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)


from xgboost import XGBClassifier

Xgb=XGBClassifier()
model=Xgb.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,auc,roc_curve

xgb_fpr,xgb_tpr,threshold=roc_curve(Y_test,Y_pred)

print(auc(xgb_fpr,xgb_tpr))
print(accuracy_score(Y_test,Y_pred))

# First I tried with default parameters and understood that the model is good still I will just alter with max_depth and get the best parameters from randomsearch


params={
        "max_depth" : [2,3],
        "colsample_bylevel": [0.1,0.5,1],
        "learning_rate" :[0.1,0.2,0.30],
        "min_child_weight": [1,5,15],
        "n_estimators":[50,100,150]
        }

from sklearn.model_selection import RandomizedSearchCV

Xgb=XGBClassifier()

random_search=RandomizedSearchCV(Xgb,param_distributions=params,n_iter=5)

random_search.fit(X_train,Y_train)

random_search.best_estimator_
random_search.best_params_

Xgb=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.5,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=3,
              min_child_weight=5, monotone_constraints='()',
              n_estimators=150, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

model=Xgb.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,auc,roc_curve

xgb_fpr,xgb_tpr,threshold=roc_curve(Y_test,Y_pred)

print(auc(xgb_fpr,xgb_tpr))
print(accuracy_score(Y_test,Y_pred)) # 80%- Accuracy



# Step 5 Working on test data- Considering XGBOOST as optimum model

Xgb=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.5,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=3,
              min_child_weight=5, monotone_constraints='()',
              n_estimators=150, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

model=Xgb.fit(X_train,Y_train)

X_test=test

# Now predicting for orginal test data

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)


Y_pred=model.predict_proba(X_test)

data=Y_pred

df=pd.DataFrame(data=data[0:,0:])


df[0].value_counts()


df=df[0]
submision=pd.concat([submission,df],axis=1)

submission=submision.loc[0:100]
submission=submission.set_index("member_id")

submission.to_csv("E:\\interview Assignments\\ML_Artivatic_dataset\\Final\\Imbalanced Dataset\\final_submission.csv")



