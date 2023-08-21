#!/usr/bin/env python
# coding: utf-8

# <h1>Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data preparation" data-toc-modified-id="Data preparation-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data preparation</a></span></li><li><span><a href="#Data analysis" data-toc-modified-id="Data analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data analysis</a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Model</a></span></li>

# # Recovery of gold from ore

# We are going to prepare a prototype of a machine learning model for the "Digit". The company develops solutions for the efficient operation of industrial enterprises.
# 
# The model should predict the recovery rate of gold from gold-bearing ore. We will use data with mining and cleaning parameters.
# 
# The model will help optimize production to launch an enterprise with profitable characteristics.
# 
# This is our work plan:
# 
# 1. Preparing the data
# 2. Conducting a research analysis of the data
# 3. Building and training a model

# # Data preparation

# Let's start by importing the necessary packages.

# In[2]:


import collections
import numpy as np
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.dummy import DummyRegressor
import warnings 
warnings.filterwarnings("ignore")


# Let's see what's in there with a loop.

# In[5]:


data_full = pd.read_csv(r'C:\Users\pinos\Downloads\gold_industry_full(1).csv')
data_train = pd.read_csv(r'C:\Users\pinos\Downloads\gold_industry_train(1).csv')
data_test = pd.read_csv(r'C:\Users\pinos\Downloads\gold_industry_test(1).csv')


# In[3]:


for i in data_full, data_train, data_test:
    data_full.info()
    data_full.head()
    data_full.describe()
    data_train.info()
    data_train.head()
    data_test.info()
    data_test.head()
    display(i)


# Most of the data is float type, minus the date, which is a string. Taking into account the type of data, we can assume that the classification model will not be the most suitable for solving the problem we are facing.
# 
# We check if there is any missing data with a loop again.

# In[4]:


for i in data_full, data_train, data_test:
    display(data_full.isna().mean())
    display(data_train.isna().mean())
    display(data_test.isna().mean())


# There is very little missing data, so we can delete it without fear that it will affect the analysis.

# In[5]:


for i in data_full, data_train, data_test:
    data_full=data_full.dropna(axis=0)
    data_train=data_train.dropna(axis=0)
    data_test=data_test.dropna(axis=0)


# Now we are going to check if everything is allright.

# In[6]:


for i in data_full, data_train, data_test:
    display(data_full.isna().any )
    display(data_train.isna().any )
    display(data_test.isna().any )


# Let's check the correlation between variables in each of the datasets.

# In[6]:


sns.heatmap(data_full.corr())
plt.title('Correlation with complete data')
plt.show()


# In[7]:


sns.heatmap(data_train.corr())
plt.title('Correlation with train data')
plt.show()


# In[8]:


sns.heatmap(data_test.corr())
plt.title('Correlation with test data')
plt.show()


# The variables are correlated.

# In[10]:


data_test.info()


# Let's check that the recovery efficiency is calculated correctly.

# In[11]:


recovery_efficiency = (data_train['rougher.output.concentrate_au'] * (data_train['rougher.input.feed_au'] - data_train['rougher.output.tail_au'])) / (
data_train['rougher.input.feed_au'] * (data_train['rougher.output.concentrate_au'] - data_train['rougher.output.tail_au'])) * 100
print(recovery_efficiency)


# In[12]:


print(data_train['rougher.output.recovery'])


# As we can see, the data match.
# 
# Let's calculate it on the training sample for the attribute rougher.output.recovery and we find the average absolute error.

# In[13]:


mean_absolute_error(recovery_efficiency, data_train['rougher.output.recovery'])


# The average absolute error is not very high, which indicates that there is no big difference, and therefore the indicators are correct.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Оценили `MAE` между исходным и расчётным значением эффективности обогащения и убедились, что эффективность обогащения рассчитана правильно - отлично!
# </div>

# In[14]:


print(data_train.shape)
print(data_test.shape)
print(data_full.shape)


# In the tested data, we see that there are fewer columns than in the rest, because they are measured and calculated much later.
# 
# We check the list of variables that we have in datasets.

# In[15]:


step = collections.Counter()
type_step = collections.Counter()
data = collections.Counter()
list_variables = [i for i in data_train.columns if i not in data_test.columns]
for i in list_variables:
    words = list(map(str,i.split(".")))
    step[words[0]] += 1
    type_step[words[1]] += 1
    data[words[2]] += 1
display(list_variables)   
display(type_step, step, data)          


# The following are not available in the target sample: rougher.calculation, final.output.recovery, final.output.concentrate_au, rougher.output.tail_sol, rougher.output.recovery and final.output.recovery.

# # Conclusions
# 
# At this first stage of data manipulation, we found out what these types are, and also discovered that there are missing values that we had to deal with. We also checked with the help of correlations how some data correlate with others. Then we prepared the data for the training stage and discarded the column that is not suitable for analyzing and creating a machine learning model. The last step was one of the most important at this stage and consisted in verifying that the recovery efficiency was calculated correctly.

# # Data Analysis

# Let's see how the concentration of metals (Au, Ag, Pb) changes at various stages of purification.

# In[9]:


phase_one_ag = data_full['rougher.input.feed_ag']
phase_two_ag = data_full['rougher.output.concentrate_ag']
phase_three_ag = data_full['primary_cleaner.output.concentrate_ag']
final_phase_ag = data_full['final.output.concentrate_ag']
plt.figure(figsize=(20,10))
plt.hist(phase_one_ag, color='red', alpha=0.3, label='Silver concentration at the first stage')
plt.hist(phase_two_ag, color='blue', alpha=0.3, label='Silver concentration at the second stage')
plt.hist(phase_three_ag, color='yellow', alpha=0.3, label='Silver concentration at the third stage')
plt.hist(final_phase_ag, color='green', alpha=0.3, label='Silver concentration at the final stage')
plt.legend()
plt.xlabel('Values')                       
plt.ylabel('Frequency')
plt.title('Silver concentration at different stages')
plt.show()


# In[13]:


phase_one_pb = data_full['rougher.input.feed_pb']
phase_two_pb = data_full['rougher.output.concentrate_pb']
phase_three_pb = data_full['primary_cleaner.output.concentrate_pb']
final_phase_pb = data_full['final.output.concentrate_pb']
plt.figure(figsize=(20,10))
plt.hist(phase_one_pb, color='red', alpha=0.3, label='Lead concentration at the first stage')
plt.hist(phase_two_pb, color='blue', alpha=0.3, label='Lead concentration at the second stage')
plt.hist(phase_three_pb, color='yellow', alpha=0.3, label='Lead concentration at the third stage')
plt.hist(final_phase_pb, color='green', alpha=0.3, label='Lead concentration at the final stage')
plt.legend()
plt.xlabel('Values')                       
plt.ylabel('Frequency')
plt.title('Lead concentration at different stages')
plt.show()


# In[14]:


phase_one_au = data_full['rougher.input.feed_pb']
phase_two_au = data_full['rougher.output.concentrate_pb']
phase_three_au = data_full['primary_cleaner.output.concentrate_pb']
final_phase_au = data_full['final.output.concentrate_pb']
plt.figure(figsize=(20,10))
plt.hist(phase_one_au, color='red', alpha=0.3, label='Gold concentration at the first stage')
plt.hist(phase_two_au, color='blue', alpha=0.3, label='Gold concentration at the second stage')
plt.hist(phase_three_au, color='yellow', alpha=0.3, label='Gold concentration at the third stage')
plt.hist(final_phase_au, color='green', alpha=0.3, label='Gold concentration at the final stage')
plt.legend()
plt.xlabel('Values')                       
plt.ylabel('Frequency')
plt.title('Gold concentration at different stages')
plt.show()


# In the first phase the silver concentration goes from less concentration to more concentration.
# 
# In the case of gold, the process is the other way around, the distribution is smaller in the first phase but becomes higher in the last phase.
# 
# Regarding lead, the concentrations increase in each of the phases, from less concentration to more concentration.
# 
# Gold at the final stage grew four times more than at the first stage, while silver fell and lead rose slightly.
# 
# The concentration of gold increased by 37 percent at the final stage, while the concentration of lead remained almost unchanged, and the concentration of silver decreased by 60 percent.
# 
# Let's now compare the size distributions of raw material in the training and test samples.

# In[12]:


plt.figure(figsize=(20,10))
sns.kdeplot(data_train['rougher.input.feed_size'], shade=True, label='Training data')
sns.kdeplot(data_test['rougher.input.feed_size'], shade=True, label='Test data')
plt.legend()
plt.title('Distribution of granules before cleaning')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.xlim(0, 200)
plt.show()


# In[20]:


plt.figure(figsize=(20,10))
sns.kdeplot(data_train['primary_cleaner.input.feed_size'], shade=True, label='Training data')
sns.kdeplot(data_test['primary_cleaner.input.feed_size'], shade=True, label='Test data')
plt.title('Distribution of granules before cleaning')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.xlim(0, 10)
plt.show()


# As we can see, the distributions are very similar, so it won't cause us any problems in the future, the data is correct.

# Now we are going to add the concentrations up at each of the stages.

# In[19]:


data_full['sum_rough'] = data_full[['rougher.output.concentrate_ag', 
                           'rougher.output.concentrate_pb',
                          'rougher.output.concentrate_sol',
                          'rougher.output.concentrate_au']].sum(axis=1)
print(data_full.sum_rough)


# In[18]:


data_full['sum_raw'] = data_full[['rougher.input.feed_ag',
                                    'rougher.input.feed_pb',
                                    'rougher.input.feed_sol',
                                    'rougher.input.feed_au']].sum(axis=1)
display(data_full.sum_raw)


# In[17]:


data_full['sum_clean'] = data_full[['primary_cleaner.output.concentrate_ag', 
                                    'primary_cleaner.output.concentrate_pb',
                                   'primary_cleaner.output.concentrate_sol',
                                   'primary_cleaner.output.concentrate_au']].sum(axis=1)
display(data_full.sum_clean)


# In[16]:


data_full['sum_final'] = data_full[['final.output.concentrate_ag',
                                   'final.output.concentrate_pb',
                                   'final.output.concentrate_sol',
                                   'final.output.concentrate_au']].sum(axis=1)
display(data_full.sum_final)


# In[20]:


ax1=data_full['sum_rough']
ax2=data_full['sum_raw']
ax3=data_full['sum_clean']
ax4=data_full['sum_final']
plt.figure(figsize=(20, 10))
sns.kdeplot(data=ax1, shade=True, label='Rudimentary')
sns.kdeplot(data=ax2, shade=True, label='Raw')
sns.kdeplot(data=ax3, shade=True, label='Clean')
sns.kdeplot(data=ax4, shade=True, label='Final')
plt.legend(loc='upper left')
plt.title('Sum of concentrated components at different stages')
plt.xlabel('Concentration')
plt.ylabel('Frequency')
plt.show()


# A higher concentration of components is detected both at the purification stage and at the final stage.
# 
# There are few elements in the range from zero to 40, the best thing we could do in this case is to remove them so that they do not affect the analysis.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Исследована суммарная концентрация металлов на разных стадиях техпроцесса - отлично! Также отмечено наличие нулевых значений и необходимость их удаления - тут всё круто!
# </div>

# In[21]:


data_full = data_full[data_full.sum_rough >= 40]
data_full = data_full[data_full.sum_clean >= 40]
data_full = data_full[data_full.sum_raw >= 40]
data_full = data_full[data_full.sum_final >= 40]


# In[22]:


plt.figure(figsize=(20, 10))
ax = data_full.sum_rough.hist(alpha=0.5, legend=True, density=True)
data_full.sum_clean.hist(ax=ax, alpha = 0.8, legend=True, density=True)
data_full.sum_raw.hist(ax=ax, legend=True, alpha=0.6, density=True)
data_full.sum_final.hist(ax=ax, legend=True, alpha=0.5, density=True)
plt.title('Sum of concentrated components at different stages')
plt.xlabel('Concentration')
plt.ylabel('Frequency')
plt.show()


# Now we see much more clearly how data is normally distributed.
# 
# At the cleaning stage, we see how the sum of the components decreases after deleting atypical values.

# ## Conclusions
# In this analytical part, we studied how the components change during extaction gold proccesing at various stages, and saw how lead and silver reduce their presence in each of them until the final phase, when the concentration of gold exceeds all the others, as expected. At the first stage, gold makes up 50% of the total number of components, silver-30% and lead-20%. At the second stage, the share of gold exceeds 60%, while the share of silver and lead is almost the same-about 20%. At the last stage, most of the silver turns into gold, which grows by 37%, while lead remains at a level similar to the previous one.
# We also studied the distribution of the components and found that the data surrounding it is distributed normally, which guarantees us the correctness of the data we are dealing with. In the part where we sum all the components, we remove the extreme values and decide to remove them so that they do not have a harmful effect on the analysis and subsequent construction of the model.

# # Model

# Let's write a function to calculate the final sMAPE.

# In[28]:


def smape(target, predicted):
    error = np.abs(target - predicted)
    scale = (np.abs(target) + np.abs(predicted)) / 2
    return np.mean(error / scale) * 100
def final_smape(smape_rougher, smape_final):
    return (0.25*smape_rougher + 0.75*smape_final)


# We prepare training data before training.

# In[29]:


data_train = data_train.drop([
 'final.output.concentrate_pb',
 'final.output.concentrate_sol',
 'final.output.concentrate_au',
 'final.output.recovery',
 'final.output.tail_ag',
 'final.output.tail_pb',
 'final.output.tail_sol',
 'final.output.tail_au',
 'primary_cleaner.output.concentrate_ag',
 'primary_cleaner.output.concentrate_pb',
 'primary_cleaner.output.concentrate_sol',
 'primary_cleaner.output.concentrate_au',
 'primary_cleaner.output.tail_ag',
 'primary_cleaner.output.tail_pb',
 'primary_cleaner.output.tail_sol',
 'primary_cleaner.output.tail_au',
 'rougher.calculation.sulfate_to_au_concentrate',
 'rougher.calculation.floatbank10_sulfate_to_au_feed',
 'rougher.calculation.floatbank11_sulfate_to_au_feed',
 'rougher.calculation.au_pb_ratio',
 'rougher.output.concentrate_ag',
 'rougher.output.concentrate_pb',
 'rougher.output.concentrate_sol',
 'rougher.output.concentrate_au',
 'rougher.output.recovery',
 'rougher.output.tail_ag',
 'rougher.output.tail_pb',
 'rougher.output.tail_sol',
 'rougher.output.tail_au',
 'secondary_cleaner.output.tail_ag',
 'secondary_cleaner.output.tail_pb',
 'secondary_cleaner.output.tail_sol',
 'secondary_cleaner.output.tail_au'], axis=1)


# In[44]:


target_train_rougher = data_train['rougher.output.recovery'].reset_index(drop=True) 


# In[45]:


target_train_final = data_train['final.output.recovery'].reset_index(drop=True) 


# In[46]:


data_features_train = data_train.drop(['date', 'rougher.output.recovery', 'final.output.recovery'], axis=1) 


# In[47]:


features_train_rougher = data_train.loc[:,data_features_train.columns] 


# In[48]:


features_train_final = features_train_rougher


# In[38]:


data_test = data_test.merge(data_full[['date', 'rougher.output.recovery', 'final.output.recovery']], on=('date'))
target_test_rougher = data_test['rougher.output.recovery'].reset_index(drop=True) 
target_test_final = data_test['final.output.recovery'].reset_index(drop=True) 
data_features = data_test.drop(['date', 'rougher.output.recovery', 'final.output.recovery'], axis=1) 
features_test_rougher = data_test.loc[:,data_features.columns] 
features_test_final = features_test_rougher


# Now we are doing the same thing, at least to create a goal in two steps.

# In[39]:


model_forest = RandomForestRegressor(random_state=1234, max_depth=10, n_estimators=50)
model_tree = DecisionTreeRegressor(random_state=1234)


# Using the GridSearchCV method, we will search for the best parameters for our models.

# In[40]:


params_random = {'n_estimators':range(10, 60, 10), 'max_depth':[None] + [i for i in range(2, 11)]} 
params_tree = {'max_depth': [None] + [i for i in range(2, 11)]}


# In[41]:


scorer = make_scorer(smape, greater_is_better=False) 


# In[42]:


grid_rougher_forest = GridSearchCV(model_forest,
                                   param_grid = params_random,
                                   scoring=scorer, n_jobs=-1, verbose=10, cv=3)
grid_final_forest = GridSearchCV(model_forest,
                                 param_grid = params_random,
                                 scoring=scorer, n_jobs=-1, verbose=10, cv=3)
grid_rougher_tree = GridSearchCV(model_tree,
                                 param_grid = params_tree,
                                 scoring=scorer, n_jobs=-1, verbose=10, cv=3)
grid_final_tree = GridSearchCV(model_tree,
                               param_grid = params_tree,
                               scoring=scorer, n_jobs=-1, verbose=10, cv=3) 


# In[43]:


grid_rougher_forest.fit(features_train_rougher,target_train_rougher)
grid_final_forest.fit(features_train_final,target_train_final)
grid_rougher_tree.fit(features_train_rougher,target_train_rougher)
grid_final_tree.fit(features_train_final, target_train_final)


# In[49]:


print("Лучшие параметры для моделей")
print(grid_rougher_forest.best_estimator_)
print(grid_final_forest.best_estimator_)
print(grid_rougher_tree.best_estimator_)
print(grid_final_tree.best_estimator_)


# In[50]:


print('Фаза более грубая')
print('Forest: ', -grid_rougher_forest.best_score_)
print('Tree: ', -grid_rougher_tree.best_score_)
print('Заключительный этап')
print('Forest: ', -grid_final_forest.best_score_)
print('Tree: ', -grid_final_tree.best_score_)


# In[51]:


print('Средневзвешенный лес: ', -final_smape(grid_rougher_forest.best_score_, grid_final_forest.best_score_))
print('Средневзвешенное дерево: ', -final_smape(grid_rougher_tree.best_score_, grid_final_tree.best_score_))


# We see that the random forest tree model gives us slightly fewer errors than the tree model, so we are going to test the forest model at the testing stage.

# Both at one stage and at another, the best video model is the random forest model, at the first stage with a big difference, and at the second stage with a much smaller difference.
# Now we are going to move on to testing a model with a random forest with the most appropriate parameters.

# In[52]:


model_forest_predict_rough = grid_rougher_forest.best_estimator_.predict(features_test_rougher)
model_forest_predict_final = grid_final_forest.best_estimator_.predict(features_test_final)


# In[53]:


smape_rougher_forest = smape(target_test_rougher, model_forest_predict_rough)
smape_final_forest = smape(target_test_final, model_forest_predict_final)


# In[54]:


print('final_smape на тестовой выборке: ', final_smape(smape_rougher_forest, smape_final_forest))


# In[55]:


dummy_rough = DummyRegressor(strategy='mean')
dummy_final = DummyRegressor(strategy='mean')
dummy_rough.fit(features_train_rougher,target_train_rougher)
dummy_final.fit(features_train_final,target_train_final)
dummy_rough_predict = dummy_rough.predict(features_test_rougher)
dummy_final_predict = dummy_final.predict(features_test_final)
smape_rougher_dummy = smape(target_test_rougher, dummy_rough_predict)
smape_final_dummy = smape(target_test_final, dummy_final_predict)
print('final_smape dummy: ', final_smape(smape_rougher_dummy, smape_final_dummy))


# Smaper for a constant model is less convenient than the random forest model, because its value is too large.

# ### Conclusions
# At the last stage, we created several instances of a decision tree regressor and a random forest regressor. We selected the best parameters for each of them and tested them, after which we found that the random forest model is better than the tree model. During testing, we got a rather low final result, so we can be sure that our model is suitable for predicting gold production and, thus, will save the company money on future production.
