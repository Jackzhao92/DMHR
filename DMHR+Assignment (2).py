
# coding: utf-8

# In[1]:


get_ipython().system(u' pip install pandasql')


# In[ ]:


#Github URL: https://github.com/Jackzhao92/DMHR


# In[2]:


import pandas as pd
import numpy as np
from pandasql import PandaSQL

get_ipython().magic(u'matplotlib inline')
import random
import matplotlib.pyplot as plt


# In[3]:


#QUESTION A
#1  Identify all GP practices and produce a table with the total number of prescriptions 
#   and their total actual cost (using the ACT COST column).

url_prac_dt = 'https://s3.eu-west-2.amazonaws.com/dmhr-data/practices_Dec2015.csv'
url_pres_dt = 'https://s3.eu-west-2.amazonaws.com/dmhr-data/prescribing_Dec2015.csv'


# In[4]:


chunksize = 10 ** 6


# In[5]:


prac_dt = pd.read_csv(url_prac_dt, header=None)

tp = pd.read_csv(url_pres_dt,low_memory=False, chunksize=chunksize, iterator=True)
pres_dt = pd.concat(tp, ignore_index=True)


# In[6]:


#There are some white space in the column names, we need to get rid of them

list(pres_dt)


# In[7]:


pres_dt.columns = [x.strip() for x in pres_dt.columns]


# In[8]:


list(pres_dt)


# In[9]:


pres_dt.head(5)


# In[10]:


#Give the practice table column names in order to select the columns more easily, and drop the last column, 
#which did not give any information

prac_dt.columns = ['period','practice','clinic_name','address','road','city','county','postcode','dontknow']
prac_dt.drop('dontknow', axis=1, inplace=True)
prac_dt.head(5)


# In[11]:


#There are some white space in the column values, we need to get rid of them

prac_dt['postcode'] = prac_dt['postcode'].map(lambda x: x.strip())
prac_dt['city'] = prac_dt['city'].map(lambda x: x.strip())
prac_dt['county'] = prac_dt['county'].map(lambda x: x.strip())


# In[12]:


#remember to check the missing value

#There is a mis-placement problem in the dataset, e.g. some city names have been put into the county column. 
#Thus I choose both city name and county name columns to match the name 'Bournemouth'
#Also, I assume that there is no spelling mistakes of the word 'Bournemouth' in the dataset
#I searched online, and I find that SOUTHBOURNE BOURNEMOUTH is also a part of Bournemouth, thus I count this city as Bournemouth

prac_BM_dt = prac_dt.loc[prac_dt['city'].str.contains('BOURNEMOUTH')|prac_dt['county'].str.contains('BOURNEMOUTH')]
prac_BM_dt
#The following table is all practices in the Bournemouth city


# In[13]:


#I do not need all practices' addresses to repeat so many times, 
#thus I pick the list of practice code in Bournemouth, and use the code list to select all prescribing data in Bournemouth

pres_BM_dt = pres_dt[pres_dt['PRACTICE'].isin(prac_BM_dt['practice'])]
pres_BM_dt.head(5)


# In[14]:


#After I check the unique value in the BNF CODE column and the BNF NAME column, I find that
#there are more unique values in the BNF CODE column than the BNF NAME column. However the number of BNF CODE should 
#be same as BNF NAME. Thus, there are some wrong value in these two columns. Since the number of unique values of BNF NAME
#is smaller than the number of unique values of BNF CODE, there are some BNF NAMEs have serval BNF CODE.
#I believe the BNF CODE column involved more incorrect data, and
#I pick the BNF NAME as the main value to identify different prescriptions.


# In[15]:


len(pd.unique(pres_BM_dt['BNF NAME']))


# In[16]:


len(pd.unique(pres_BM_dt['BNF CODE']))


# In[17]:


len(pd.unique(pres_dt['BNF CODE']))


# In[18]:


len(pd.unique(pres_dt['BNF NAME']))


# In[19]:


#total number of prescriptions, since the problem did not define 
#the total number of prescriptions in item level or quantity leve, 
#I choose to use item level as the total number of total number of prescriptions.
#I sum up all items for each prescription

pres_BM_temp1 = pres_BM_dt.groupby(['BNF NAME'],as_index=False)['ITEMS'].sum()
pres_BM_temp1.head(5)


# In[21]:


#Sum up actual cost for each prescription
pres_BM_temp2 = pres_BM_dt.groupby(['BNF NAME'],as_index=False)['ACT COST'].sum()
pres_BM_temp2.head(5)


# In[22]:


#The table with the total number of prescriptions and their total actual cost

pres_BM_TABLE = pd.merge(pres_BM_temp1,pres_BM_temp2,on='BNF NAME')

#Change the columns' name for the table
pres_BM_TABLE.columns = ['BNF NAME', 'total number of prescriptions', 'total actual cost']
pres_BM_TABLE


# In[23]:


#QUESTION A
#2  Find the top ten most/least frequently prescribed medications across all practices. 
#   What is their total actual cost and how does that compare to 
#   the overall actual costs of each practice and of the entire city?


# In[24]:


#Since the problem did not define frequently prescribed medications in the item level or in the quantity level.
#I choose to use item level as the standard to define frequently prescribed medications.
#TOP 10 most frequently prescribed medications.

pres_most_freq_each_total_cost = pres_BM_TABLE.sort_values('total number of prescriptions', ascending=False).head(10)
pres_most_freq_each_total_cost


# In[25]:


#Top 10 least frequently prescribed medications
#All least ten frequently prescribed medications' ITEM numbers are all equal to 1, 
#it is possible some prescribed medications also have ITEM numbers equal to 1, but since the 
#order to top ten least frequently prescribed medications arranged in alphabetical order, 
#their name did not show on the list.

pres_least_freq_each_total_cost = pres_BM_TABLE.sort_values('total number of prescriptions', ascending=True).head(10)
pres_least_freq_each_total_cost


# In[26]:


#overall actual costs of each practice in Bournemouth
prac_pres_BM_TABLE = pres_BM_dt.groupby(['PRACTICE'],as_index=False)['ACT COST'].sum()
prac_pres_BM_TABLE 


# In[27]:


#overall actual costs of the entire city
entire_city_cost = prac_pres_BM_TABLE['ACT COST'].sum()
entire_city_cost


# In[28]:


#Top 10 most frequently prescribed medications total actual cost
pres_most_freq_each_total_cost['total actual cost'].sum()


# In[29]:


#Percentage of the actual cost of top ten most frequently prescribed medications across all practices in the Bournemouth
#in the acutal cost of entire city is 3.2%
top_ten_freq_perc_BM = pres_most_freq_each_total_cost['total actual cost'].sum()/entire_city_cost
top_ten_freq_perc_BM


# In[30]:


#Percentage of the actual cost of least ten frequently prescribed medications across all practices in the Bournemouth
#in the acutal cost of entire city is 0.0099%
least_ten_freq_perc_BM = pres_least_freq_each_total_cost['total actual cost'].sum()/entire_city_cost
least_ten_freq_perc_BM


# In[31]:


#The total actual cost of top ten most frequently prescribed medications compare to 
#the overall actual costs of each practice in the Bournemouth

#Although the percentage of the actual cost of top ten most frequently prescribed medications across all practices in the Bournemouth
#in the acutal cost of entire city is only 3.2%, there are 10 practices' actual cost is 
#less than the total actual cost of top ten most frequently prescribed medications

len(prac_pres_BM_TABLE[prac_pres_BM_TABLE['ACT COST'] <= pres_most_freq_each_total_cost['total actual cost'].sum()])


# In[32]:


#The total actual cost of top ten least frequently prescribed medications compare to 
#the overall actual costs of each practice in the Bournemouth

#There are only 2 practices' actual cost is less than the total actual cost of top ten least frequently prescribed medications

len(prac_pres_BM_TABLE[prac_pres_BM_TABLE['ACT COST'] <= pres_least_freq_each_total_cost['total actual cost'].sum()])


# In[33]:


#The average actual cost for each practice is 93024.33, which is more than the total actual cost of 
#top ten most frequently prescribed medications across all practices in the Bournemouth, 
#since the total actual cost of top ten most frequently prescribed medications is 83580.98

#In addition, the median value of actual cost for each practice is 104225.69 which is also higher than the 
#total actual cost of the top ten most frequently prescribed medications and 
#the top ten least frequently prescribed medications


prac_pres_BM_TABLE['ACT COST'].mean()


# In[34]:


prac_pres_BM_TABLE['ACT COST'].median()


# In[35]:


#The standard deviation of top 10 most frequently prescribed medications actual cost is 5124.3, and for 
#top 10 least frequently prescribed medications actual cost is 32.47
#Both values are much lower than the standard deviation of the total actual cost of all practices in the Bournemouth.

pres_most_freq_each_total_cost['total actual cost'].std()


# In[36]:


pres_least_freq_each_total_cost['total actual cost'].std()


# In[37]:


prac_pres_BM_TABLE['ACT COST'].std()


# In[38]:


#QUESTION A
#3  Find the top ten most expencive medications and calculate their total actual cost.


#Since the problem did not define the word 'expensive' in item level or in quantity level, in this question
#I choose to use quantity level expensive (total cost/quantity) as the standard to compare each medication.


pres_each_quantity_sum = pres_BM_dt.groupby(['BNF NAME'],as_index=False)['QUANTITY'].sum()
pres_quantity_cost_table = pd.merge(pres_each_quantity_sum, pres_BM_temp2, 
                                           on='BNF NAME', how='inner')
pres_quantity_cost_table


# In[39]:


#Define expensive means comparing the cost per drug, rather than item
#top ten most expencive medications and their total actual cost
pres_quantity_cost_table['cost_per_drug'] = pres_quantity_cost_table['ACT COST']/pres_quantity_cost_table['QUANTITY']
pres_expensive_sort = pres_quantity_cost_table.sort_values('cost_per_drug', ascending=False)
pres_expensive_sort.head(10)


# In[40]:


#bar chart of top 10 expensive med here


# In[41]:


pres_tp10_expensive = pres_expensive_sort.head(10)
ax=pres_tp10_expensive[['BNF NAME','cost_per_drug']].plot(kind='bar', x='BNF NAME', y='cost_per_drug', title='top ten most expencive medications in Bournemouth', figsize=(6,4),legend=True, fontsize=10)

ax.set_xlabel("BNF",fontsize=8)

ax.set_ylabel("cost per drug",fontsize=8)


# In[42]:


ax=pres_tp10_expensive[['BNF NAME','ACT COST']].plot(kind='bar', x='BNF NAME', y='ACT COST', title='top ten most expencive medications in Bournemouth', figsize=(6,4),legend=True, fontsize=10)

ax.set_xlabel("BNF",fontsize=8)

ax.set_ylabel("Actual cost",fontsize=8)


# In[44]:


#QUESTION A
#4  How does prescribing (frequency and costs) in your city 
#   compare when using prescribing data from Cambridge as a reference?

#prescribing (frequency and costs) in Bournemouth is the following table
pres_BM_TABLE




# In[45]:


#Since there are other cities' names contain the name of 'CAMBRIDGE', thus I use exact match method to find out all practices
#in the CAMBRIDGE
prac_CB_dt = prac_dt.loc[(prac_dt['city'] == 'CAMBRIDGE') | (prac_dt['county'] == 'CAMBRIDGE')]
prac_CB_dt.head(5)


# In[46]:


pres_CB_dt = pres_dt[pres_dt['PRACTICE'].isin(prac_CB_dt['practice'])]
pres_CB_dt.head()


# In[47]:


#as I mentioned above, I choose to use BNF NAME and ITEM columns.


# In[48]:


pres_CB_total_med_freq_sum = pres_CB_dt.groupby(['BNF NAME'],as_index=False)['ITEMS'].sum()

pres_CB_total_med_freq_sum.head(5)


# In[49]:


pres_CB_total_med_cost_sum = pres_CB_dt.groupby(['BNF NAME'],as_index=False)['ACT COST'].sum()
pres_CB_total_med_cost_sum.head(5)


# In[52]:


#total number of prescriptions and their total actual cost in Cambridge 
pres_CB_freq_cost_table = pd.merge(pres_CB_total_med_freq_sum, 
                                   pres_CB_total_med_cost_sum, on='BNF NAME', how='inner')
pres_CB_freq_cost_table


# In[53]:


#total number of prescriptions and their total actual cost in Bournemouth 
pres_BM_TABLE


# In[54]:


#comparison on city level

#overall actual costs of Cambridge is 2717047.69
CB_entire_city_cost = pres_CB_freq_cost_table['ACT COST'].sum()
CB_entire_city_cost


# In[55]:


#overall actual costs of Bournemouth is 2604681.28, which is slightly less than Cambridge's
BM_entire_city_cost = entire_city_cost
BM_entire_city_cost


# In[56]:


#Total actual cost of prescribing in Bournemouth is 95.9% of total actual cost of prescribing in Cambridge, 
#We can say that these two cities has similar total actual cost of prescribing.

perc = BM_entire_city_cost/CB_entire_city_cost
perc


# In[57]:


#The mean of actual cost of prescribing in Cambridge is higher than in Bournemouth


# In[58]:


#The mean of actual cost of prescribing in Cambridge
CB_pres_cost_mean = pres_CB_freq_cost_table['ACT COST'].mean()
CB_pres_cost_mean


# In[59]:


#The mean of actual cost of prescribing in Bournemouth
BM_pres_cost_mean = pres_BM_TABLE['total actual cost'].mean()
BM_pres_cost_mean


# In[60]:


#The standard deviation of actual cost of prescribing in Cambridge is higher than in Bournemouth


# In[61]:


#The standard deviation of actual cost of prescribing in Cambridge
CB_pres_cost_std = pres_CB_freq_cost_table['ACT COST'].std()
CB_pres_cost_std


# In[62]:


#The standard deviation of actual cost of prescribing in Bournemouth
BM_pres_cost_std = pres_BM_TABLE['total actual cost'].std()
BM_pres_cost_std


# In[63]:


#The total number of prescriptions in Cambridge is higher than in Bournemouth


# In[64]:


#The total number of prescriptions in Cambridge
CB_pres_total_num = pres_CB_freq_cost_table['ITEMS'].sum()
CB_pres_total_num


# In[65]:


#The total number of prescriptions in Bournemouth
BM_pres_total_num = pres_BM_TABLE['total number of prescriptions'].sum()
BM_pres_total_num


# In[66]:


#comparsion of frequency in both cities


# In[67]:


#Top 10 most frequently prescribed medications in Bournemouth
BM_top_ten_freq_pres = pres_BM_TABLE.sort_values('total number of prescriptions', ascending=False)
BM_top_ten_freq_pres.head(10)


# In[68]:


#Top 10 most frequently prescribed medications in Cambridge 
CB_top_ten_freq_pres = pres_CB_freq_cost_table.sort_values('ITEMS', ascending=False)
CB_top_ten_freq_pres.head(10)


# In[69]:


#Common top 10 most frequently prescribed medications
common_med = pd.merge(CB_top_ten_freq_pres.head(10), BM_top_ten_freq_pres.head(10), on='BNF NAME', how='inner')
common_med


# In[70]:


#Compare the top ten frequently prescribed medications in Bournemouth and Cambridge, 
#we can find that there are 6 frequently prescribed medications are the same. 
#Omeprazole_Cap E/C 20mg, Aspirin Disper_Tab 75mg, Simvastatin_Tab 40mg
#Paracet_Tab 500mg, Amlodipine_Tab 5mg, Salbutamol_Inha 100mcg (200 D) CFF, Bendroflumethiazide_Tab 2.5mg.
#In addition, the frequency of prescribing these medication in Cambridge are all higher than in Bournemouth.


# In[71]:


#Top 10 least frequently prescribed medications in Bournemouth
BM_least_ten_freq_pres = pres_BM_TABLE.sort_values('total number of prescriptions', ascending=True)
BM_least_ten_freq_pres.head(10)


# In[72]:


#Top 10 least frequently prescribed medications in Cambridge 
CB_least_ten_freq_pres = pres_CB_freq_cost_table.sort_values('ITEMS', ascending=True)
CB_least_ten_freq_pres.head(10)


# In[73]:


#Common Top 10 least frequently prescribed medications
common_med_least_freq = pd.merge(CB_least_ten_freq_pres.head(10), 
                                 BM_least_ten_freq_pres.head(10), on='BNF NAME', how='inner')
common_med_least_freq


# In[74]:


#Compare the least ten frequently prescribed medications in the Bournemouth and Cambridge,
#we can find that all least frequently prescribed medications are only prescribed once time.
#And there is no common least ten frequently prescribed medications in the Bournemouth and Cambridge.
#However, although we did not find common least ten frequently prescribed medications in the Bournemouth and Cambridge,
#it is possible that this result is due to the selection process. All least ten frequently prescribed medications' ITEM
#numbers are all equal to 1, it is possible some prescribed medications also have ITEM numbers equal to 1, but since the 
#order to top ten least frequently prescribed medications arranged in alphabetical order, their name did not show on the 
#list.


# In[75]:


common_med_least_freq_adj_ver = pd.merge(
    BM_least_ten_freq_pres[BM_least_ten_freq_pres['total number of prescriptions'] == 1],
    CB_least_ten_freq_pres[CB_least_ten_freq_pres['ITEMS'] == 1], on='BNF NAME', how='inner')
common_med_least_freq_adj_ver


# In[76]:


#As a result, we can find that there are 225 common least ten frequently prescribed medications 
#in the Bournemouth and Cambridge, and their items number are all equal to 1


# In[77]:


#QUESTION A
#5  Using SQL, produce a table that provides the number of GP practices per city, ordered in descending order.

pdsql = PandaSQL()


# In[78]:


#deal with missing-data problem.
#After searching for the missing values in city, I found that many rows did not have values for city is because 
#the values of city have been filled into the county column, thus I want to fix this problem

prac_dt[(prac_dt['city'] == "")]


# In[79]:


#I filled county value into city column if the city column is empty, 
#and delete the rows where city and county are both empty.

prac_dt.loc[(prac_dt['city'] == ""), 'city'] = prac_dt[(prac_dt['city'] == "")]['county']


# In[80]:


prac_dt.drop(prac_dt[(prac_dt['city'] == "")].index, inplace=True)


# In[81]:


#sql

sql_query = "SELECT city, COUNT(practice) FROM prac_dt GROUP BY city ORDER BY COUNT(practice) DESC"
print (pdsql(sql_query, locals()))


# In[82]:


#Use pandas to check if the result is matched

a = prac_dt.groupby(['city'],as_index=False)['practice'].count()
b = a.sort_values('practice', ascending=False)
b.head(10)


# In[83]:


#QUESTION B
#1  Calculate the monthly total spending for each GP-practice.

url_ONS_pc = 'https://s3.eu-west-2.amazonaws.com/dmhr-data/postcodes.csv'
url_gp_demogra = 'https://digital.nhs.uk/media/28273/Numbers-of-Patients-Registered-at-a-GP-Practice-Jan-2016-GP-Practice-and-quinary-age-groups/Any/gp-reg-patients-prac-quin-age'


# In[84]:


chunksize2 = 10 ** 5
tp2 = pd.read_csv(url_ONS_pc,low_memory=False, chunksize=chunksize2, iterator=True)
ons_pos = pd.concat(tp2, ignore_index=True)


# In[85]:


ons_pos.head(5)
#list(ons_pos)


# In[86]:


gp_demogra = pd.read_csv(url_gp_demogra)


# In[87]:


gp_demogra.head(5)


# In[88]:


pres_dt.head(5)


# In[89]:


#monthly total spending for each GP-practice
#Since the dataset is from Dec 2015, only one month, thus the monthly total spending for each GP-practice only need to 
#groupby each practice and sum up their actual cost.

monthly_total_spend = pres_dt.groupby(['PRACTICE'],as_index=False)['ACT COST'].sum()
monthly_total_spend


# In[ ]:


#QUESTION B
#2  Use the number of registered patients in each GP-practice to calculate the relative costs per patient.


# In[90]:


#we need to make sure two tables have the same values in all GP-practice code, 
#thus, in the GP demographic table, I selected all practices which are also in the monthly total spending table as the 
#practices for this problem.
practice_list_in_demogra = gp_demogra[gp_demogra['GP_PRACTICE_CODE'].isin(monthly_total_spend['PRACTICE'])]
practice_list_in_demogra.head(5)


# In[91]:


#Many columns are not useful for this question, thus I choose to drop those columns

practice_list_in_demogra.columns


# In[92]:


dropcol = [u'ONS_CCG_CODE', u'CCG_CODE',
       u'ONS_REGION_CODE', u'NHSE_REGION_CODE', u'ONS_COMM_RGN_CODE',
       u'NHSE_COMM_REGION_CODE', u'Male_0-4', u'Male_5-9', u'Male_10-14', u'Male_15-19', u'Male_20-24',
       u'Male_25-29', u'Male_30-34', u'Male_35-39', u'Male_40-44',
       u'Male_45-49', u'Male_50-54', u'Male_55-59', u'Male_60-64',
       u'Male_65-69', u'Male_70-74', u'Male_75-79', u'Male_80-84',
       u'Male_85-89', u'Male_90-94', u'Male_95+', u'Female_0-4', u'Female_5-9',
       u'Female_10-14', u'Female_15-19', u'Female_20-24', u'Female_25-29',
       u'Female_30-34', u'Female_35-39', u'Female_40-44', u'Female_45-49',
       u'Female_50-54', u'Female_55-59', u'Female_60-64', u'Female_65-69',
       u'Female_70-74', u'Female_75-79', u'Female_80-84', u'Female_85-89',
       u'Female_90-94', u'Female_95+']


# In[93]:


prac_patient_num = practice_list_in_demogra.drop(dropcol, axis=1)


# In[94]:


prac_patient_num.head(10)


# In[95]:


#I choose to inner join the monthly total spending table and the patient number of each practice table
prac_patient_spend_table = pd.merge(monthly_total_spend, prac_patient_num, 
                                    left_on='PRACTICE', right_on='GP_PRACTICE_CODE', how='inner')
prac_patient_spend_table.drop('GP_PRACTICE_CODE', axis=1, inplace=True)
prac_patient_spend_table.head(10)


# In[96]:


#Calculate the cost per patient by using monthly total spending of each practice 
#divided by total number of patients of each practice 
prac_patient_spend_table['cost_per_patient'] = prac_patient_spend_table['ACT COST']/prac_patient_spend_table['Total_All']


# In[97]:


#Use the number of registered patients in each GP-practice to calculate the relative costs per patient.
prac_patient_spend_table


# In[98]:


#QUESTION B
#3  Visualize the monthly total spending per registered patients for all GP-practices in a scatterplot, 
#   show a trend line, and visualize the data for your city within the national scatterplot


# In[99]:


#Select all practice in Bournemouth
BM_prac_patient_spend = prac_patient_spend_table[prac_patient_spend_table['PRACTICE'].isin(prac_BM_dt['practice'])]
BM_prac_patient_spend


# In[100]:


#The max value for cost per patient is too large.
#There is an outlier problem in cost per patient row, we need to deal with it.

prac_patient_spend_table.cost_per_patient.max()


# In[101]:


#The way I used to deal with outlier is the following 
prac_patient_spend_table['cost_per_patient'].quantile(.996)


# In[102]:


#The blue points represented to the monthly total spending per registered patients for all GP-practices
#The red line is the trend line
#The yellow points represented to the monthly total spending per registered patients for Bournemouth.



from math import floor

fig=plt.figure(figsize=(8, 6))
ax=fig.add_subplot(111)
x= prac_patient_spend_table['cost_per_patient']
y = prac_patient_spend_table['Total_All']
w = BM_prac_patient_spend['cost_per_patient']
z = BM_prac_patient_spend['Total_All']
ax.scatter(x, y, color='royalblue')
ax.scatter(w,z, color='yellow')
fit = np.polyfit(x, y, deg=1)
ax.plot(x, fit[0] * x + fit[1], color='red')
ax.set_title('Scatter plot: monthly total spending / registered patients')
ax.set_xlabel("monthly total spending per registered patients",fontsize=12)
ax.set_ylabel("total number of registered patients",fontsize=12)
max_x = floor(prac_patient_spend_table.cost_per_patient.quantile(.996))
max_y = floor(prac_patient_spend_table.Total_All.quantile(.996))
ax.set_xlim(0, max_x)
ax.set_ylim(0, max_y)




# In[100]:


#QUESTION B
#4  Visualize the relative costs per patient of all national GP-practices in a histogram.  


# In[103]:



x = prac_patient_spend_table.cost_per_patient.values

plt.hist(x, bins=100000)
plt.xlabel("Monthly prescription spending per patient")
plt.ylabel("Frequency")

min_x = floor(prac_patient_spend_table['cost_per_patient'].quantile(.01))
max_x = floor(prac_patient_spend_table['cost_per_patient'].quantile(.99))
plt.xlim(min_x, max_x) #do not show outliers
plt.title("Relative costs per patient of all national GP-practices")
plt.show()


# In[ ]:


#QUESTION B
#5  Use descriptive statistics to show how your assigned city compares to the national level.


# In[104]:


BM_prac_patient_spend.head(5)


# In[105]:


prac_patient_spend_table.head(5)


# In[106]:


#percentage of patients. From the result, we can see that the total number of patients in Bournemouth is only 0.37% 
#of national total number of patients, which is quite small amount of people.


# In[110]:


BM_patient_number = float(BM_prac_patient_spend.Total_All.sum())
BM_patient_number


# In[111]:


Nation_patient_number = float(prac_patient_spend_table.Total_All.sum())
Nation_patient_number


# In[109]:


BM_patient_perc = (BM_patient_number/Nation_patient_number)*100
BM_patient_perc


# In[112]:


#Compare the mean of cost per patient in Bournemouth and national level.
#First, we directly calculate the mean of cost per patient in both levels, 
#and we can see that the mean of cost per patient in national level is much higher than in the Bournemouth.
#The mean of cost per patient in national level is 21.32 and in the Bournemouth is only 12.36


# In[113]:


BM_cost_per_patient_ave = float(BM_prac_patient_spend.cost_per_patient.mean())
BM_cost_per_patient_ave


# In[114]:


Nation_cost_per_patient_ave = float(prac_patient_spend_table.cost_per_patient.mean())
Nation_cost_per_patient_ave


# In[115]:


cost_per_patient_compar = BM_cost_per_patient_ave/Nation_cost_per_patient_ave
cost_per_patient_compar


# In[116]:


#However, there are some outliers in the national level dataset, thus I chose to control the influence of outlier, 
#and then compare the 'true' mean of cost per patient in both levels. 
#After getting rid of outliers in the national level, we can see that the mean of cost per patient in national level is 
#similar to the mean in the Bournemouth.
#The adjust mean of cost per patient in national level is 13.04, 
#which is similar to the mean of cost per patient in the Bournemouth.


# In[117]:


wo_outlier_nation_patient_spending = prac_patient_spend_table[prac_patient_spend_table['cost_per_patient'] <= 
                                                              prac_patient_spend_table['cost_per_patient'].quantile(.996)]
wo_outlier_Nation_cost_per_patient_ave = float(wo_outlier_nation_patient_spending.cost_per_patient.mean())
wo_outlier_Nation_cost_per_patient_ave


# In[118]:


BM_cost_per_patient_ave


# In[119]:


wo_outlier_cost_per_patient_compar = BM_cost_per_patient_ave/wo_outlier_Nation_cost_per_patient_ave
wo_outlier_cost_per_patient_compar


# In[120]:


#The standard deviation of costs per patient of all national GP-practices is higher than in Bournemouth


# In[121]:


BM_cost_per_patient_std = float(BM_prac_patient_spend.cost_per_patient.std())
BM_cost_per_patient_std


# In[122]:


wo_outlier_Nation_cost_per_patient_std = float(wo_outlier_nation_patient_spending.cost_per_patient.std())
wo_outlier_Nation_cost_per_patient_std


# In[119]:


#The median value of costs per patient of all national GP-practices is higher than in Bournemouth


# In[123]:


BM_cost_per_patient_med = float(BM_prac_patient_spend.cost_per_patient.median())
BM_cost_per_patient_med


# In[124]:


wo_outlier_Nation_cost_per_patient_med = float(wo_outlier_nation_patient_spending.cost_per_patient.median())
wo_outlier_Nation_cost_per_patient_med


# In[125]:


#quartile range for cost per patient in national level is 25% 10.68, 50% 13.19, 75% 15.46, which is similar to in the Bournemouth
#In the Bournemouth, quartile range is 25% 10.9, 50% 12.17, 75% 14.05, 


# In[126]:


wo_outlier_nation_patient_spending.describe()


# In[127]:


BM_prac_patient_spend.describe()


# In[128]:


#QUESTION C
#1  Identify for all GP-practices the relative costs per patient for all statin prescriptions 
#   (simvastatin, atorvastatin, rosuvastatin, pravastatin, fluvastatin) by using the dataset from December 2015.


# In[129]:


url_english_deprivation_indices = 'https://s3.eu-west-2.amazonaws.com/dmhr-data/deprivation-by-postcode.csv'
url_CVD_mortality_rate = 'https://s3.eu-west-2.amazonaws.com/dmhr-data/NHSOF_1.1_I00656_D.csv'


# In[130]:


edi_dt = pd.read_csv(url_english_deprivation_indices)
edi_dt['Postcode'] = edi_dt['Postcode'].map(lambda x: x.strip())
edi_dt.head(10)


# In[131]:


CVD_MR_dt = pd.read_csv(url_CVD_mortality_rate)
CVD_MR_dt.head(10)


# In[132]:


#Nystatin is also a kind of statin ?

#Select all statin prescriptions
pres_statin_dt = pres_dt.loc[pres_dt['BNF NAME'].str.contains('statin')]
pres_statin_dt.head(5)


# In[133]:


#Calculate total actual cost of statin for each practice
pres_statin_total_cost = pres_statin_dt.groupby(['PRACTICE'],as_index=False)['ACT COST'].sum()
pres_statin_total_cost.head(5)


# In[134]:


#Get a table of practices which have statin prescriptions, and the total number of patients of those practices
prac_statin_patient_spend_table = pd.merge(pres_statin_total_cost, prac_patient_num, 
                                    left_on='PRACTICE', right_on='GP_PRACTICE_CODE', how='inner')
prac_statin_patient_spend_table.drop('GP_PRACTICE_CODE', axis=1, inplace=True)
prac_statin_patient_spend_table.head(5)


# In[135]:


#Calculate relative costs per patient for all statin prescriptions
prac_statin_patient_spend_table['cost_per_patient'] = prac_statin_patient_spend_table['ACT COST']/prac_statin_patient_spend_table['Total_All']


# In[136]:


#all GP-practices the relative costs per patient for all statin prescriptions

prac_statin_patient_spend_table


# In[134]:


#QUESTION C
#2  Identify for all GP-practice the associated Index of Multiple Deprivation (IMD) 
#   for each GP-Practice in your assigned city.


# In[95]:


#list(edi_dt)


# In[146]:


#Since the problem did not told us to select all old and new practice, or just still live practice,
#I decided to select all Bournemouth practices in the English deprivation indices dataset with the condition 
#that the postcode status is Live in order to get rid of those old practice data with old postcode.
#I believe the data of live practice is more useful to do the analysis.

#If the problem want us to use all practice, just delete the condition (edi_dt['Postcode Status'] == 'Live')
#and directly use pd.merge to inner join the edi_dt and prac_BM_dt, then drop the useless columns.


prac_IMD_BM = edi_dt[edi_dt['Postcode'].isin(prac_BM_dt['postcode']) & (edi_dt['Postcode Status'] == 'Live')]
prac_IMD_BM


# In[139]:


#There are too many columns in the table which are not useful for this problem, thus I choose to delete those columns
list(prac_IMD_BM)


# In[140]:


dropcol2 = [
 'LSOA code',
 'LSOA Name',
 'Income Rank',
 'Income Decile',
 'Income Score',
 'Employment Rank',
 'Employment Decile',
 'Employment Score',
 'Education and Skills Rank',
 'Education and Skills Decile',
 'Health and Disability Rank',
 'Health and Disability Decile',
 'Crime Rank',
 'Crime Decile',
 'Barriers to Housing and Services Rank',
 'Barriers to Housing and Services Decile',
 'Living Environment Rank',
 'Living Environment Decile',
 'IDACI Rank',
 'IDACI Decile',
 'IDACI Score',
 'IDAOPI Rank',
 'IDAOPI Decile',
 'IDAOPI Score']


# In[141]:


#Select IMD for each Bournemouth practices' postcodes

prac_only_IMD_dt = prac_IMD_BM.drop(dropcol2, axis=1)
prac_only_IMD_dt.head(5)


# In[142]:


#Link the Bournemouth practice postcode to Bournemouth practice

prac_IMD_BM_final_table = pd.merge(prac_BM_dt, prac_only_IMD_dt, 
                                    left_on='postcode', right_on='Postcode', how='inner')
prac_IMD_BM_final_table.drop(['postcode'], axis=1, inplace=True)
prac_IMD_BM_final_table


# In[143]:


#In the table, I find that there are some practices have been duplicated 2 times, 
#and I want all unique practices appeared only once.


# All GP-practice the associated Index of Multiple Deprivation (IMD) for each GP-Practice in Bournemouth
IMD_BM_prac_real_final = prac_IMD_BM_final_table.drop_duplicates(subset=['practice'])
IMD_BM_prac_real_final


# In[147]:


#QUESTION C
#3  Use the entire national dataset and identify the lowest relative spenders of statins from the first decile 
#   and the highest relative spenders of statins from the last decile. 
#   Now determine for all identified GP-practices for both groups (lowest and the highest) 
#   the associated Index of Multiple Deprivation (IMD). Use these two groups to assess whether the IMD-score differs. 
#   Use descriptive statistics for your answer.


# In[148]:


#The quantile 0-10 is first decile, the quantile 90-100 is last decile


# Another way to select first decile and last decile is using following command
#pd.qcut(prac_statin_patient_spend_table.cost_per_patient,10)


# In[149]:


#the lowest relative spenders of statins from the first decile crude table

Statin_lowest_spender_crude = prac_statin_patient_spend_table[
    prac_statin_patient_spend_table['cost_per_patient'] 
    <= prac_statin_patient_spend_table['cost_per_patient'].quantile(0.1)]
Statin_lowest_spender_crude


# In[150]:


#the highest relative spenders of statins from the last decile crude table

Statin_highest_spender_crude = prac_statin_patient_spend_table[
    prac_statin_patient_spend_table['cost_per_patient'] 
    > prac_statin_patient_spend_table['cost_per_patient'].quantile(0.9)]
Statin_highest_spender_crude


# In[151]:


Statin_lowest_spender_mid = pd.merge(edi_dt, Statin_lowest_spender_crude, 
                                     left_on='Postcode', right_on='POSTCODE', how='inner')
list(Statin_lowest_spender_mid)


# In[152]:


dropcol9 = [
 'LSOA code',
 'LSOA Name',
 'Income Rank',
 'Income Decile',
 'Income Score',
 'Employment Rank',
 'Employment Decile',
 'Employment Score',
 'Education and Skills Rank',
 'Education and Skills Decile',
 'Health and Disability Rank',
 'Health and Disability Decile',
 'Crime Rank',
 'Crime Decile',
 'Barriers to Housing and Services Rank',
 'Barriers to Housing and Services Decile',
 'Living Environment Rank',
 'Living Environment Decile',
 'IDACI Rank',
 'IDACI Decile',
 'IDACI Score',
 'IDAOPI Rank',
 'IDAOPI Decile',
 'IDAOPI Score',
 'ACT COST',
 'POSTCODE',
 'Total_Male',
 'Total_Female']


# In[153]:


Statin_lowest_spender_mid.drop(dropcol9, axis=1, inplace=True)


# In[154]:


Statin_lowest_spender_mid.head()


# In[155]:


#After merge two tables, I found that there is a duplicated data problem. For instance, in Statin_lowest_spender table, 
#there only one record of postcode 'TS18 1HU', but after the table merged with EDI table, there are 4 record for this postcode


Statin_lowest_spender_mid[Statin_lowest_spender_mid['Postcode'] == 'TS18 1HU']


# In[156]:


Statin_lowest_spender_crude[Statin_lowest_spender_crude['POSTCODE'] == 'TS18 1HU']


# In[157]:


#I find that in the English deprivation indices table, 
#there are some practices have exactly same information, and it will cause duplicated data problem during the inner join
#process. I believe this problem happened is because the English deprivation indices table assume that the practices in 
#the same postcode have same statistics.
#Since the same postcode have same statistics, I only need to left one record for each postcode in EDI table.
#thus I choose to remove these duplicated practice by using postcode


#It is reasonable that there are some practices share one postcode since some practices can be in the same building 
edi_dt[edi_dt['Postcode'] == 'TS18 1HU']


# In[158]:


edi_dt_adj = edi_dt.drop_duplicates(subset=['Postcode'])

Statin_lowest_spender_final = pd.merge(edi_dt_adj, Statin_lowest_spender_crude, 
                                     left_on='Postcode', right_on='POSTCODE', how='inner')

Statin_lowest_spender_final.drop(dropcol9, axis=1, inplace=True)


# In[159]:


#GP-practices for lowest groups the associated Index of Multiple Deprivation (IMD)
Statin_lowest_spender_final


# In[160]:


#Same thing happened in the highest group

Statin_highest_spender_final = pd.merge(edi_dt_adj, Statin_highest_spender_crude, 
                                     left_on='Postcode', right_on='POSTCODE', how='inner')

Statin_highest_spender_final.drop(dropcol9, axis=1, inplace=True)


# In[161]:


#GP-practices for highest groups the associated Index of Multiple Deprivation (IMD)

#The final Statin_highest_spender_final table has one low less than the Statin_highest_spender_crude table.
#I think the reason is that there is one practice, which did not be included in the EDI table, thus there is one missing.

Statin_highest_spender_final


# In[ ]:


#Use these two groups to assess whether the IMD-score differs. Use descriptive statistics for your answer.


# In[162]:


#Compare the mean of Index of Multiple Deprivation Rank in two groups, we can find that the mean of high group
#is higher than the low group, the mean of highest group is 14026.26 while the mean of lowest group is 12252.90

high_IMD_mean = Statin_highest_spender_final['Index of Multiple Deprivation Rank'].mean()
high_IMD_mean


# In[163]:


low_IMD_mean = Statin_lowest_spender_final['Index of Multiple Deprivation Rank'].mean()
low_IMD_mean


# In[164]:


#Then we compare the median of Index of Multiple Deprivation Rank in two groups, we can find that the median value of high group
#is higher than the low group. The median of highest group is 12933 while the median of lowest group is 9965

high_IMD_med = Statin_highest_spender_final['Index of Multiple Deprivation Rank'].median()
high_IMD_med


# In[165]:


low_IMD_med = Statin_lowest_spender_final['Index of Multiple Deprivation Rank'].median()
low_IMD_med


# In[166]:


#Then we compare the max value of Index of Multiple Deprivation Rank in two groups, we can find that the max value of high group
#is similar to the low group, the max of highest group is 32701 while the max of lowest group is 32837


# In[167]:


high_IMD_max = Statin_highest_spender_final['Index of Multiple Deprivation Rank'].max()
high_IMD_max


# In[168]:


low_IMD_max = Statin_lowest_spender_final['Index of Multiple Deprivation Rank'].max()
low_IMD_max


# In[169]:


#Then we compare the min value of Index of Multiple Deprivation Rank in two groups, we can find that the min value of high group
#is much higher than the low group.
high_IMD_min = Statin_highest_spender_final['Index of Multiple Deprivation Rank'].min()
high_IMD_min


# In[170]:


low_IMD_min = Statin_lowest_spender_final['Index of Multiple Deprivation Rank'].min()
low_IMD_min


# In[171]:


#Then we compare the standard deviation of Index of Multiple Deprivation Rank in two groups, 
#we can find that the standard deviation of high group is slightly higher than the low group
high_IMD_std = Statin_highest_spender_final['Index of Multiple Deprivation Rank'].std()
high_IMD_std


# In[172]:


low_IMD_std = Statin_lowest_spender_final['Index of Multiple Deprivation Rank'].std()
low_IMD_std


# In[173]:


#I calculate the 95% confidence interval for the difference between both groups. The 95% CI did not contain 0, 
#and the lower bound is much higher than 0, which indicates that there is a different in the mean of two groups' IMD score

import math
n1 = len(Statin_highest_spender_final['Index of Multiple Deprivation Rank'])
n2 = len(Statin_lowest_spender_final['Index of Multiple Deprivation Rank'])
m1 = high_IMD_mean
m2 = low_IMD_mean
s1 = high_IMD_std
s2 = low_IMD_std

high_low_IMD_CI = [
    (m1-m2)-1.96*math.sqrt((s1**2/n1)+(s2**2/n2)),
    (m1-m2)+1.96*math.sqrt((s1**2/n1)+(s2**2/n2))]

high_low_IMD_CI


# In[174]:


#According to the previous descriptive statistics result, I can say that the IMD-score is different in two groups


# In[175]:


#t-test
#the p-value nearly equals 0.00014, which indicate that the IMD score is different in two groups


# In[176]:


from scipy.stats import ttest_ind


# In[177]:


ttest_ind(Statin_highest_spender_final['Index of Multiple Deprivation Rank'], Statin_lowest_spender_final['Index of Multiple Deprivation Rank'])


# In[ ]:


#QUESTION C
#4  Identify for all GP-practices the associated nine English regions. 
#   Identify for each region the associated 75 mortality rate for cardiovascular diseases for the year 2015.


# In[178]:


#First, we need to decide whether we should use region name or region code in order to identify 9 regions
pd.unique(ons_pos['Region Code'])


# In[179]:


pd.unique(ons_pos['Region Name'])


# In[ ]:


#As a result, we can see there are more noise value in region code, thus I choose to use region name to identify 9 regions


# In[180]:


pd.unique(ons_pos['Country Name'])


# In[181]:


#Delete the columns which are not useful in this problem.

list(ons_pos)


# In[182]:




dropcol3 = [
 'Date Introduced',
 'User Type',
 'Easting',
 'Northing',
 'Positional Quality',
 'County Code',
 'County Name',
 'Local Authority Code',
 'Local Authority Name',
 'Ward Code',
 'Ward Name',
 'Country Code',
 'Country Name',
 'Parliamentary Constituency Code',
 'Parliamentary Constituency Name',
 'European Electoral Region Code',
 'European Electoral Region Name',
 'Primary Care Trust Code',
 'Primary Care Trust Name',
 'Lower Super Output Area Code',
 'Lower Super Output Area Name',
 'Middle Super Output Area Code',
 'Middle Super Output Area Name',
 'Output Area Classification Code',
 'Output Area Classification Name',
 'Longitude',
 'Latitude',
 'Spatial Accuracy',
 'Last Uploaded',
 'Location',
 'Socrata ID']


# In[183]:


ons_pos_left_pc_region = ons_pos.drop(dropcol3, axis=1)
ons_pos_left_pc_region.head(5)


# In[184]:


#There are some NaN in region name, I deleted those rows which contains NaN in Region Name column

ons_pos_left_pc_region_woNa = ons_pos_left_pc_region.dropna(subset = ['Region Name'])
ons_pos_left_pc_region_woNa.head(5)


# In[185]:


#all GP-practices the associated nine English regions.
prac_asso_nine_region = pd.merge(prac_dt, ons_pos_left_pc_region_woNa, 
                                    left_on='postcode', right_on='Postcode 1', how='inner')

prac_asso_nine_region.drop(['postcode'], axis=1, inplace=True)
prac_asso_nine_region


# In[186]:


#In order to identify for each region the associated 75 mortality rate for cardiovascular diseases for the year 2015
#I take a look of the Under 75 CVD mortality rates dataset, and find the 9 region information contains in the level,
#level description and Breakdown columns


#When the Breakdown column equal to 'Region' value, the dataset will show the information on the region level, 
#and each of the 9 regions will be noted in level and level description columns.


CVD_MR_dt.head()


# In[187]:


pd.unique(CVD_MR_dt['Breakdown'])


# In[188]:


pd.unique(CVD_MR_dt['Level'])


# In[189]:


pd.unique(CVD_MR_dt['Level description'])


# In[190]:


#Thus I choose to Year = 2015 in order to find 2015's 75 mortality rate for cardiovascular diseases
#choose Breakdown = 'Region' to find the information in the region level
#And choose Gender = person to show the overall statistics rather than showing the information in males or females
#Since the question only ask to identify for each region the associated 75 mortality rate for cardiovascular diseases for the year 2015.


CVD_MR_9_reg_table = CVD_MR_dt[(CVD_MR_dt['Year'] == 2015) & (CVD_MR_dt['Breakdown'] == 'Region') & (CVD_MR_dt['Gender'] == 'Person')]
CVD_MR_9_reg_table


# In[191]:


#This is the information for each region the associated 75 mortality rate for cardiovascular diseases for the year 2015
#With male and female data.
CVD_MR_dt[(CVD_MR_dt['Year'] == 2015) & (CVD_MR_dt['Breakdown'] == 'Region')]


# In[192]:


#QUESTION C
#5   Visualize (using matplotlib) for each region spending for statins [x-axis] and the mortality rate [y-axis]. 
#    Assess whether relative spending for statin prescriptions in each regions correlates with the mortality rate 
#    from cardiovascular diseases.


# In[193]:


#practice in 9 region
prac_asso_nine_region.head()


# In[194]:


#spending for statins in each practice
prac_statin_patient_spend_table.head()


# In[195]:


#mortality rate
CVD_MR_9_reg_table


# In[196]:


#First, I choose to inner join the statin patient spend table and the practices associated with 9 regions table
#in order to find for each practice in 9 region, what is their actual cost for statin.


prac_act_cost_nine_reg = pd.merge(prac_statin_patient_spend_table, prac_asso_nine_region, 
                                    left_on='PRACTICE', right_on='practice', how='inner')
prac_act_cost_nine_reg


# In[197]:


#deleted the columns that are not useful for this problem

list(prac_act_cost_nine_reg)


# In[198]:


dropcol4=[
 'Total_Male',
 'Total_Female',
 'cost_per_patient',
 'practice',
 'address',
 'road',
 'city',
 'county',
 'Postcode 1',
 'Postcode 2',
 'Postcode 3']


# In[199]:


prac_act_cost_nine_reg_drop = prac_act_cost_nine_reg.drop(dropcol4, axis=1)
prac_act_cost_nine_reg_drop.head()


# In[200]:


#get the total actual cost for statins in each region
nine_reg_total_cost = prac_act_cost_nine_reg_drop.groupby(['Region Name'],as_index=False)['ACT COST'].sum()
nine_reg_total_cost


# In[201]:


#get the total patients number in each region
nine_reg_total_patient = prac_act_cost_nine_reg_drop.groupby(['Region Name'],as_index=False)['Total_All'].sum()
nine_reg_total_patient


# In[202]:


nine_reg_cost_patient_final_table = pd.merge(nine_reg_total_patient, nine_reg_total_cost, on='Region Name', how='inner')
nine_reg_cost_patient_final_table


# In[203]:


#For each region, the patient number is different, thus it is not reasonable to directly compare the total cost for statins
#but the cost per patient is much reasonable.

#Calculate cost per patient
nine_reg_cost_patient_final_table['COST PER PATIENT'] = nine_reg_cost_patient_final_table['ACT COST']/nine_reg_cost_patient_final_table['Total_All']
nine_reg_cost_patient_final_table


# In[204]:


#Inner join the 75 mortality rate for CVD table and the cost per patient table

nine_reg_cost_per_pat_MR = pd.merge(CVD_MR_9_reg_table, nine_reg_cost_patient_final_table, 
                                    left_on='Level description', right_on='Region Name', how='inner')
nine_reg_cost_per_pat_MR


# In[205]:


nine_reg_cost_per_pat_MR['Indicator value'] = nine_reg_cost_per_pat_MR['Indicator value'].astype(float)


# In[206]:


nine_reg_cost_per_pat_MR_after_adj = nine_reg_cost_per_pat_MR.sort_values('COST PER PATIENT', ascending=False)


# In[207]:


nine_reg_cost_per_pat_MR_after_adj2 = nine_reg_cost_per_pat_MR.sort_values('ACT COST', ascending=False)


# In[208]:


get_ipython().magic(u'matplotlib inline')

ax0 = nine_reg_cost_per_pat_MR_after_adj.plot(kind = 'bar', x = 'COST PER PATIENT', y = 'Indicator value', 
                                    title = 'The relation of spending for statin prescriptions in each regions and the mortality rate from cardiovascular diseases',
                                   stacked=False)
ax0.legend(['Region'])


# In[209]:


ax0 = nine_reg_cost_per_pat_MR_after_adj2.plot(kind = 'bar', x = 'ACT COST', y = 'Indicator value', 
                                    title = 'The relation of spending for statin prescriptions in each regions and the mortality rate from cardiovascular diseases',
                                   stacked=False)
ax0.legend(['Region'])


# In[210]:


ax1 = nine_reg_cost_per_pat_MR_after_adj.plot(kind = 'line', x = 'COST PER PATIENT', y = 'Indicator value', 
                                    title = 'The relation of spending for statin prescriptions in each regions and the mortality rate from cardiovascular diseases',
                                   stacked=False)
ax1.legend(['Region'])


# In[211]:


ax1 = nine_reg_cost_per_pat_MR_after_adj2.plot(kind = 'line', x = 'ACT COST', y = 'Indicator value', 
                                    title = 'The relation of spending for statin prescriptions in each regions and the mortality rate from cardiovascular diseases',
                                   stacked=False)
ax1.legend(['Region'])


# In[212]:


import matplotlib.pyplot as plt
fig=plt.figure(figsize=(8, 6))
ax=fig.add_subplot(111)
x = nine_reg_cost_per_pat_MR_after_adj['COST PER PATIENT']
y = nine_reg_cost_per_pat_MR_after_adj['Indicator value']
ax.scatter(x, y, color='royalblue')
fit = np.polyfit(x, y, deg=1)
ax.plot(x, fit[0] * x + fit[1], color='red')
ax.set_title('The relation of spending for statin prescriptions in each regions and the mortality rate from CVD')
ax.set_xlabel("Cost per patient for statins",fontsize=12)
ax.set_ylabel("mortality rate",fontsize=12)


# In[213]:


#According to the bar chart and the line chart, the pattern of positive relation is not that obvious.
#If we just see the result of those two chart, I do not think the relative spending for statin prescriptions 
#in each regions correlates with the mortality rate from cardiovascular diseases.
#However, after fit a trend line, we can see it seems there is a positive relation between cost per patient for statins and the mortality rate
#The cost per patient for statins is higher, the mortality rate is higher. 
#But the trend line may also easily be influenced since there is too less data point.


# In[ ]:


#QUESTION 4
#1  Provide a visualisation of the seasonal patterns across all years.


# In[214]:


url_au = 'https://www.google.org/flutrends/about/data/flu/au/data.txt'

#set header=8 to properly extract the data
australiaFluTrends = pd.read_csv(url_au, sep=',', header = 8)
australiaFluTrends['Date'] = pd.to_datetime(australiaFluTrends['Date'])


# In[215]:


australiaFluTrends.head()


# In[216]:


url_ca = 'https://www.google.org/flutrends/about/data/flu/ca/data.txt'

#set header=8 to properly extract the data
canadaFluTrends = pd.read_csv(url_ca, sep=',', header=8)
canadaFluTrends['Date'] = pd.to_datetime(canadaFluTrends['Date'])


# In[217]:


canadaFluTrends.head()


# In[218]:


get_ipython().magic(u'matplotlib inline')
#Plot and store the flu trens for Canada

ax = canadaFluTrends.plot(legend ='left', x='Date', y = 'Canada', figsize=(18, 10), grid=True)
#Plot the flu trend for Australia; ax=ax plots the chart into the previous

australiaFluTrends.plot(x='Date', y = 'Australia' , ax=ax)


# In[219]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
fig_ca = canadaFluTrends.plot(x='Date', figsize=(18,10))
fig_ca.set_title('The trend of flu search activity in Canada',fontsize=15)
fig_ca.set_ylabel('Frequency',fontsize=15)
fig_ca.set_xlabel('Year',fontsize=15)


# In[220]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
fig_au = australiaFluTrends.plot(x='Date', figsize=(18,10))
fig_au.set_title('The trend of flu search activity in Australia',fontsize=15)
fig_au.set_ylabel('Frequency',fontsize=15)
fig_au.set_xlabel('Year',fontsize=15)


# In[ ]:


#Calculate the yearly minimum and maximum for each country. 
#Provide and plot a reasonable mathematical function that could be used as an approximation for the seasonal trend for each country.


# In[221]:


australiaFluTrends['Year'] = australiaFluTrends['Date'].dt.year


# In[222]:


aus_yearly_max = australiaFluTrends.groupby(['Year'],as_index=False)['Australia'].max()
aus_yearly_min = australiaFluTrends.groupby(['Year'],as_index=False)['Australia'].min()


# In[223]:


#Australia yearly minimum and maximum 
Aus_yearly_min_and_max = pd.merge(aus_yearly_max, aus_yearly_min, on='Year', how='inner')
Aus_yearly_min_and_max.columns = ['Year','Maximum','Minimum']
Aus_yearly_min_and_max


# In[224]:


canadaFluTrends['Year'] = canadaFluTrends['Date'].dt.year


# In[225]:


can_yearly_max = canadaFluTrends.groupby(['Year'],as_index=False)['Canada'].max()
can_yearly_min = canadaFluTrends.groupby(['Year'],as_index=False)['Canada'].min()


# In[226]:


#Canada yearly minimum and maximum 
Can_yearly_min_and_max = pd.merge(can_yearly_max, can_yearly_min, on='Year', how='inner')
Can_yearly_min_and_max.columns = ['Year','Maximum','Minimum']
Can_yearly_min_and_max


# In[ ]:


#Math function should follow the following format: y = x^4*b1 + x^3*b2 + x^2*b3 + x^1*b4 + b5
#I used the polyfit to get the coefficient of the function, and plot the function.


# In[227]:


from matplotlib import pyplot
from numpy import polyfit

X = [7*i%365 for i in range(0, len(australiaFluTrends['Date']))]
y = australiaFluTrends['Australia'].astype(float)
degree = 4
coef = polyfit(X, y, degree)
print('Coefficients: %s' % coef)

curve = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree-d) * coef[d]
    curve.append(value)

pyplot.plot(y)
pyplot.plot(curve, color='red', linewidth=2)
pyplot.show()


# In[ ]:


#According to the above result, for Australia, the function is 
#Y = 1.35050719e-06 * X^4 - 1.08519477e-03 * x^3 + 2.60847988e-01 * X^2 - 1.70924537e+01 * X + 3.69377196e+02 


# In[228]:


X = [7*i%365 for i in range(0, len(canadaFluTrends['Date']))]
y = canadaFluTrends['Canada'].astype(float)
degree = 4
coef = polyfit(X, y, degree)
print('Coefficients: %s' % coef)

curve = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree-d) * coef[d]
    curve.append(value)

pyplot.plot(y)
pyplot.plot(curve, color='red', linewidth=2)
pyplot.show()


# In[ ]:


#According to the above result, for Canada, the function is 
#Y = -7.77259919e-09 * X^4 + 6.05519076e-04 * x^3 - 3.46528181e-01 * X^2 + 4.72029032e+01 * X + 1.32002136e+03 


# In[ ]:


#try of time series. Not the main part of the answer, just play with the function.


# In[ ]:


#https://www.datacamp.com/community/tutorials/time-series-analysis-tutorial


# In[229]:


list(australiaFluTrends)


# In[230]:


drop_aus = [
 'Australian Capital Territory',
 'New South Wales',
 'Queensland',
 'South Australia',
 'Victoria',
 'Western Australia',
 'Year']


# In[231]:


australiaFluTrends_after_drop = australiaFluTrends.drop(drop_aus, axis=1)


# In[232]:


australiaFluTrends_after_drop = australiaFluTrends_after_drop.set_index('Date')


# In[233]:


import statsmodels.api as sm


# In[234]:


aus=sm.tsa.seasonal_decompose(australiaFluTrends_after_drop.Australia)


# In[235]:


resplot = aus.plot()


# In[236]:


list(canadaFluTrends)


# In[237]:


drop_can=[
 'Alberta',
 'British Columbia',
 'Manitoba',
 'New Brunswick',
 'Newfoundland and Labrador',
 'Nova Scotia',
 'Ontario',
 'Saskatchewan',
 'Quebec',
 'Year']


# In[238]:


canadaFluTrends_after_drop = canadaFluTrends.drop(drop_can, axis=1)


# In[239]:


canadaFluTrends_after_drop = canadaFluTrends_after_drop.set_index('Date')


# In[240]:


can=sm.tsa.seasonal_decompose(canadaFluTrends_after_drop.Canada)


# In[241]:


resplot = can.plot()


# In[ ]:


#x(t) = s(t) + m(t) + e(t)
#Where t is the time coordinate
#x is the data
#s is the seasonal component
#e is the random error term
#m is the trend


# In[410]:


aus.trend.head(2)


# In[409]:


aus.seasonal.head(2)


# In[408]:


aus.resid.head(2)


# In[ ]:


#Following part is just some notes, no the part of the answer of anything.


# In[ ]:


#set season: spring = range(3, 5) summer = range(6, 8) fall = range(9, 11) winter = c(12, 1, 2)
#1 = winter, 2 = spring, 3 = summer, 4 = fall

australiaFluTrends['season'] = (australiaFluTrends['Date'].dt.month%12 + 3)//3
canadaFluTrends['season'] = (canadaFluTrends['Date'].dt.month%12 + 3)//3


# In[394]:


def excel_date(date1):
    temp = datetime.datetime(1899, 12, 30)    
    delta = date1 - temp
    return float(delta.days) + (float(delta.seconds) / 86400)


# In[401]:


aus_time = []
for i in range(0,len(australiaFluTrends['Date'])):
    aus_time.append(excel_date(australiaFluTrends['Date'][i]))


# In[403]:


aus_time = np.array(aus_time)

