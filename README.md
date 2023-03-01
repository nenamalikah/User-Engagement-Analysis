# User-Engagement-Analysis

## Findings 
For the model I developed, account age (length of time between account creation and last login), organization id, whether or not an account’s creation source was a personal project, opting into the mailing list, and whether or not an account’s creation source was a guest invite were the five most helpful factors in predicting whether or not a user will become an adopted user. 

## Approach
I split the challenge into three main tasks: the first task was to identify who adopted users were, the second task was to identify a model to help predict which users would adopt, and the third task was to identify which features were most helpful in predicting user adoption. 

For the first task, I primarily used the engagement table. To start off, I filtered the engagement table to contain only users with three or more logins. Following that, I created a separate column to get the timestamp for a week from each login date. Finally, I created a loop to check whether the next two login timestamps for each login timestamp were within a week of the original login timestamp. Based on this, I created a new column for adopted users. 

``` 
#Creating filter to get user ids with three or more logins

login_filter = engagement.groupby('user_id')['user_id'].filter(lambda x: len(x) >= 3)
adopted_engagement = engagement[engagement['user_id'].isin(login_filter)]
```
```
#Creating a new column that contains the date for one week from each login timestamp

week = []
adopted_engagement['time_stamp'] = pd.to_datetime(adopted_engagement['time_stamp'])
for timestamp in adopted_engagement['time_stamp']:
    week.append(timestamp + datetime.timedelta(days=7))
    
adopted_engagement['week'] = week
adopted_engagement.head()
```
```
#Creating a new column for adopted users
#The loop checks that the next two timestamps for one user's login timestamp are within one week of each other

adopted_user = []

for index in range(0,201000):
    if adopted_engagement['user_id'][index] == adopted_engagement['user_id'][index+1]:
        if (adopted_engagement['time_stamp'][index+1] < adopted_engagement['week'][index]) & (adopted_engagement['time_stamp'][index+2] < adopted_engagement['week'][index]):
            adopted_user.append('yes')
        else:
            adopted_user.append('no')
    else:
        adopted_user.append('diff_user')
        
adopted_user.extend(['outside_idx','outside_idx'])
adopted_engagement['adopted_user'] = adopted_user 
adopted_engagement.head()
```

For the second task, I trained and tested four models to see how they performed on the dataset and how well they could predict an adopted user. I tried four models total: a decision tree model, a gradient boosting model, a random forest model, and a balanced random forest model (since the distribution of the target variable is imbalanced). The Balanced Random Forest model had the highest balanced accuracy score, so I moved forward with that model. 
```
#Testing four tree-based models for their average accuracy and balanced accuracy

models = []

models.append(('CART', DecisionTreeClassifier()))
models.append(('XGB', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('BRF', BalancedRandomForestClassifier()))

accuracy = []
balanced_accuracy = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_validate(model, x_train, y_train, cv=kfold, 
                            scoring=['accuracy','balanced_accuracy'],return_train_score=True)
    accuracy.append(scores['test_accuracy'])
    balanced_accuracy.append(scores['test_balanced_accuracy'])
    names.append(name)
    msg = "%s: Average Accuracy (%f); Average Balanced Accuracy (%f)" % (name, scores['test_accuracy'].mean(),scores['test_balanced_accuracy'].mean())
    print(msg)
```

For the final task, I used the Shap library to explain and provide more insight into my model’s predictions. I used the Shap library because it provides the ability to explain the model’s overall predictions as well as individual predictions. I was particularly interested in which features (and feature values) were most helpful in predicting true positives, and which features (and feature values) may have caused the model to predict false negatives. 

To achieve this, I separated the dataframe into true positives, true negatives, false positives, and false negatives. I then defined a function to help develop waterfall plots for individual predictions. I chose a waterfall plot because it also provides the feature values contained in each individual prediction. 
```
#Separating individual predictions based on if they were correct or not 

test_df = pd.concat([x_test,y_test],axis=1)
test_df['probability_adopting'] = rf.predict_proba(x_test)[:,1]
test_df['test_set_idx'] = np.arange(len(test_df))

false_negatives = test_df.query('confirmed_adopted == "1" & probability_adopting < 0.50')
true_negatives = test_df.query('confirmed_adopted == "0" & probability_adopting < 0.50')
false_positives = test_df.query('confirmed_adopted == "0" & probability_adopting > 0.50')
true_positives = test_df.query('confirmed_adopted == "1" & probability_adopting > 0.50')
```
```
def shap_explanation(df, index,label=1):
    """This function creates a waterfall plot for a given row in a specified dataframe."""
    idx = int(df.iloc[index]['test_set_idx'])
    shap.plots.waterfall(shap_test[:,:,label][idx],max_display=11)
```

## Further Research and Data 
Given further time with this project, I think it would be helpful to conduct additional feature engineering and go through a feature selection process. There are still variables that can be created from the existing data. As an example, it would be interesting to see if the time of year that someone created their account is associated with a user adopting. It might be the case that some of the existing variables are not particularly helpful in predicting whether a user will become an adopted user.

I imagine that data such as the length of time someone uses the product during each session would be a good predictor. Furthermore, more information on the current variables could be helpful as well. As an example, for individuals who have a guest invite or an organization invite, it would be interesting to see if certain categories of organizations are more associated with users adopting. 

Outside of additional data, I would test additional models and attempt to tune their parameters to see if a more precise model could be developed than the one used in this project. 
