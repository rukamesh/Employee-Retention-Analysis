
# coding: utf-8

# # Data Visualisation and Distribution

# In[1]:

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')


plt.style.use('ggplot')

blue_color =  '#7F7FFF'
red_color = '#BF3F7F'
green_color = '#9ACD32'


# In[2]:

df = pd.read_csv('HR_Data.csv')
print("Number of rows: {}".format(df.shape[0]))
print("Number of columns: {}\n".format(df.shape[1]))

print("Column Names:")
print("----------------")
for col in df.columns:
    print(col+" ("+str(df[col].dtype)+")")
print("----------------\n")

print("Any NaN values in data: " + str(df.isnull().values.any()))

df.head()


# 1. It looks like the Department column is mis-spelled as sales let's correct it and let's keep the column names in lowercase.
# 2. There are no missing values.

# In[3]:

df=df.rename(columns = {'sales':'department'})
df.columns = [x.lower() for x in df.columns]

df['left'] = df['left'].astype('str')
df['work_accident'] = df['work_accident'].astype('str')
df['promotion_last_5years'] = df['promotion_last_5years'].astype('str')

print("Column Names:")
print("----------------")
for col in df.columns:
    print(col+" ("+str(df[col].dtype)+")")
print("----------------")

df.head(8)


# # Exploring Features
# 1. We do not see any time duration for this dataset i.e. duration in which certain number of employees (tagged as 1 in left column) left the company. In order to calculate Turnover-Rate, we need to know some duration. We may assume this is one year's data.
# 2. We are not sure how Satisfaction Levels were inferred, these scores were computed somehow.
# 3. Let's look at potential target variables left & satisfaction_level

# ## Below section shows plots for Satisfaction Level & Left Status
# 1. Plotted Histogram on Satisfaction Level
# 2. Plotted Bar chart of Left Status
# 3. Filled Histogram bars of Satisfaction level with employees who left.

# In[4]:


import copy

##########################################################
# Create figure with 4 subplots
f, ax = plt.subplots(2,2,figsize=(14,14))

(ax1, ax2, ax3, ax4) = ax.flatten()

##########################################################
# Bar Chart of Left Column
left_count = df['left'].value_counts()
left_indices = left_count.index.tolist()
left_values = left_count.values.tolist()

if (left_indices[0] == '1'):
    left_indices[0] = 'Left'
    left_indices[1] = 'Stayed'
else:
    left_indices[0] = 'Stayed'
    left_indices[1] = 'Left'
    
y_pos = np.arange(len(left_values))    
bars=ax1.bar(y_pos, left_values, align='center')

bars[0].set_color(green_color)
bars[1].set_color(red_color)

# Add counts on Bars
def autolabel(rects):
    for rect in rects:
        ax1.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height()-1000,
                '%d' % int(rect.get_height()),
                ha='center', va='bottom')
autolabel(bars)

# Add Text showing percentage of employees who left
emp_left = left_values[1]
perc_left = emp_left/sum(left_values) * 100
ax1.text(0.55, 8000, "Turnover Percentage:\n {:.2f}%".format(perc_left), fontsize=11)

ax1.set_xticks(y_pos)
ax1.set_xticklabels(left_indices)
ax1.set_ylabel('Frequency')
ax1.set_title('Employees Status: Stayed vs Left')


##########################################################
# Histogram of Satisfaction Level: I want 20 bins in range (0-1)
ax2.hist(df['satisfaction_level'], bins=20, range=(0,1), alpha=0.5)
ax2.set_title('Histogram: Satisfaction Level')
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Satisfaction-Level')


##########################################################


n, bins, patches = ax3.hist(df['satisfaction_level'], bins=20, range=(0,1), alpha=0.5)
## 
left_in_bins = []
for i in range(len(bins)-1):
    start = bins[i]
    end = bins[i+1]
    
    left_emp = len(df.loc[(df['satisfaction_level']>=start) & (df['satisfaction_level']<end) & (df['left'] == '1')])
    left_in_bins.append(left_emp)


index = 0
for_legend = None
for p in patches:
    patch = copy.copy(p)
    patch.set_height(left_in_bins[index])
    #patch.set_color(red_color)
    patch.set_hatch('//')
    patch.set_alpha(1.0)
    ax3.add_patch(patch)
    if index==1:
        for_legend = patch
    index = index + 1
ax3.set_title('Histogram: Satisfaction-Level with Left-Status')
ax3.set_ylabel('Frequency')
ax3.set_xlabel('Satisfaction-Level')
ax3.legend([for_legend], ['Employees Left'])

##########################################################

n, bins, patches = ax4.hist(df['satisfaction_level'], bins=20, range=(0,1), alpha=0.5)
## 
left_in_bins = []
for i in range(len(bins)-1):
    start = bins[i]
    end = bins[i+1]
    
    left_emp = len(df.loc[(df['satisfaction_level']>=start) & (df['satisfaction_level']<end) & (df['left'] == '1')])
    left_in_bins.append(left_emp)


index = 0
for p in patches:
    patch = copy.copy(p)
    patch.set_height(left_in_bins[index])
    if index in range(3):
        patch.set_color('r')
    elif index in range(7,10):
        patch.set_color('y')
    elif index in range(14,20):
        patch.set_color('b')
        
    else:
        patch.set_color(red_color)
        patch.set_alpha(1.0)
    ax4.add_patch(patch)
    index = index + 1
ax4.set_title('Histogram:Satisfaction-Level with Left-Status Segments')
ax4.set_xlabel('Satisfaction-Level')
ax4.set_ylabel('Frequency')

plt.show()

seg_one_left = sum(left_in_bins[0:3])
print((seg_one_left/emp_left) * 100.)

seg_two_left = sum(left_in_bins[7:10])
print((seg_two_left/emp_left) * 100.)

seg_three_left = sum(left_in_bins[14:20])
print((seg_three_left/emp_left) * 100.)


# ### We can clearly see three segments or behaviors.
# 1. First bin in histogram is empty. Second and Third mostly contains people who left the company. These people must be really unhappy since their satisfaction level is below 0.15. (1st-Segment) 25.4% of total Employees who left
# 2. Then we see peaks around 0.4. (2nd-Segment) 43.8% of total Employees who left Lastly, there's a chunk 0.7-0.95 who left the company.(3rd-Segment) 26.1% of total Employees who left . That's almost 1/4th of employees who left.
# 3. 94% of Employees who left are in these 3 segments.

# ### Hypothesis
# 1. There could be multiple reasons for leaving a company and for 3rd-Segment I think satisfaction is not the criteria for leaving. There 'could be' other reasons like monthly hours, work accidents or promotion, which is not really reflected in Satisfaction Scores.
# 2. The other two segments are behaviors i.e. certain employees hitting ~0.4 satisfaction threshold are likely to churn leave. At-least these two segments show that employees are not-satisfied unlike (3rd-Segment).
# 3. Ideally, I think, Satisfaction levels should correspond to number of employees who are leaving. But it is really not the case here, for example, 3rd-Segment does not make sense.

# ## Below section shows plots for Left Status, Department & Salary

# In[5]:

import copy 



##########################################################
# Create fiure with 3 subplots
f, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(10,25))

##########################################################
# Departments Bar Chart
dept_count = df['department'].value_counts()
dept_indices = dept_count.index.tolist()
dept_values = dept_count.values.tolist()

# Employees left in certain department
emp_left = []
for dept in dept_indices:
    left_emp = len(df.loc[(df['department']==dept) & (df['left'] == '1')])
    emp_left.append(left_emp)

# Percentage of employees who left in certain department
emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,dept_values)]


y_pos = np.arange(len(dept_values))
bars=ax1.bar(y_pos, dept_values, align='center', color=blue_color,edgecolor='black')
emp_left_bars=ax1.bar(y_pos, emp_left, align='center',color=blue_color,hatch='//',edgecolor='black')

# Add counts on Bars
def autolabel(rects, ax):
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:d}".format(int(rect.get_height())),
                ha='center', va='bottom')
autolabel(bars, ax1)

# Add percentage on Bars
def autolabel_emp(rects, ax):
    index = 0
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:.0f}%".format(int(emp_left_perc[index])),
                ha='center', va='bottom')
        index = index + 1
autolabel_emp(emp_left_bars, ax1)
    

ax1.set_xticks(y_pos)
ax1.set_xticklabels(dept_indices)
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)

ax1.set_ylabel('Frequency')
ax1.set_title('Employees who left w.r.t Departments')
ax1.legend((bars[0], emp_left_bars[0]), ('Employees in Dept.', 'Employees Left'))

##########################################################
# Salary Bar Chart

sal_count = df['salary'].value_counts()
sal_indices = sal_count.index.tolist()
sal_values = sal_count.values.tolist()

# Employees left w.r.t salary
emp_left = []
for sal in sal_indices:
    left_emp = len(df.loc[(df['salary']==sal) & (df['left'] == '1')])
    emp_left.append(left_emp)

# Percentage of employees who left in certain salary range
emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,sal_values)]


y_pos = np.arange(len(sal_values))
bars=ax2.bar(y_pos, sal_values, align='center', color=blue_color,edgecolor='black')
emp_left_bars=ax2.bar(y_pos, emp_left, align='center',color=blue_color,hatch='//',edgecolor='black')

autolabel(bars,ax2)
autolabel_emp(emp_left_bars,ax2)

ax2.set_xticks(y_pos)
ax2.set_xticklabels(sal_indices)
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)

ax2.set_ylabel('Frequency')
ax2.set_title('Employees who left w.r.t Salary')
ax2.legend((bars[0], emp_left_bars[0]), ('Employees in Salary-Range.', 'Employees Left'))

##########################################################
dept_count = df['department'].value_counts()
dept_indices = dept_count.index.tolist()
dept_values = dept_count.values.tolist()

low_sal = []
med_sal = []
high_sal = []
for dept in dept_indices:
    low_sal.append(len(df.loc[(df['department']==dept) & (df['salary'] == 'low')]))
    med_sal.append(len(df.loc[(df['department']==dept) & (df['salary'] == 'medium')]))
    high_sal.append(len(df.loc[(df['department']==dept) & (df['salary'] == 'high')]))

y_pos = np.arange(len(dept_values))
low_bars=ax3.bar(y_pos, low_sal, align='center', color=red_color)
med_bars=ax3.bar(y_pos, med_sal, align='center', color=green_color,bottom=low_sal)
high_bars=ax3.bar(y_pos, high_sal, align='center', color=blue_color,bottom=np.add(low_sal, med_sal))

ax3.set_xticks(y_pos)
ax3.set_xticklabels(dept_indices)
for tick in ax3.get_xticklabels():
    tick.set_rotation(45)


ax3.set_ylabel('Frequency')
ax3.set_title('Departments w.r.t Salary')
ax3.legend((high_bars[0],med_bars[0],low_bars[0]), ('High Salary','Medium Salary','Low Salary'))

plt.show()


# ### From the first graph
# 1. Turnover rate (percentage of churned employees) of departments 'RandD' & 'Management' is lowest ~15%
#    Rest of them are roughly same, with 'HR' being the highest ~29%, when it comes to employee-churn

# ### From the second graph
# 1. Highly paid employees are less likely to leave the company, which is very intuitive.

# ### Interesting Insights/Hypothesis
# 1. 'HR' is a tough department to be in.
# 2. 'HR' is highest at employee turnover. It is the second-last department w.r.t employees strength or numbers (first graph). And within this small department, there are extremely small number of people getting high salaries (third graph), which means that the growth opportunity in 'HR' is very less and hence employees leave the company. Only 45 out of 739 'HR' employees are getting high salaries.

# ## Below section shows plots for Promotion, Work-Accidents & No. of Project

# In[6]:

##########################################################
# Create fiure with 4 subplots
f, ax = plt.subplots(2,2,figsize=(12,12))

(ax1, ax2, ax3, ax4) = ax.flatten()

##########################################################
# Work-Accident Bar Chart
acc_count = df['work_accident'].value_counts()
acc_indices = acc_count.index.tolist()
acc_values = acc_count.values.tolist()


# Employees left w.r.t Accidents
emp_left = []
for acc in acc_indices:
    left_emp = len(df.loc[(df['work_accident']==acc) & (df['left'] == '1')])
    emp_left.append(left_emp)

# Percentage of employees w.r.t Accidents
emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,acc_values)]


y_pos = np.arange(len(acc_values))
bars=ax1.bar(y_pos, acc_values, align='center', color=blue_color, edgecolor='black')
emp_left_bars=ax1.bar(y_pos, emp_left, align='center', color=blue_color, hatch='//',edgecolor='black')

#ax1.

# Add counts on Bars
def autolabel(rects, ax):
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:d}".format(int(rect.get_height())),
                ha='center', va='bottom')
autolabel(bars, ax1)

# Add percentage on Bars
def autolabel_emp(rects, ax):
    index = 0
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:.0f}%".format(int(emp_left_perc[index])),
                ha='center', va='bottom')
        index = index + 1
autolabel_emp(emp_left_bars, ax1)
    

ax1.set_xticks(y_pos)
#ax1.set_xticklabels(acc_indices)
ax1.set_xticklabels(["No Work-Accident","Work-Accident"])
ax1.set_ylabel('Frequency')
ax1.set_title('Employees who left w.r.t Accidents')
ax1.legend([emp_left_bars[0]], ['Employees Left'])

##########################################################
# Promotion Bar Chart
promo_count = df['promotion_last_5years'].value_counts()
promo_indices = promo_count.index.tolist()
promo_values = promo_count.values.tolist()


# Employees left w.r.t promotion
emp_left = []
for p in promo_indices:
    left_emp = len(df.loc[(df['promotion_last_5years']==p) & (df['left'] == '1')])
    emp_left.append(left_emp)

# Percentage of employees w.r.t promotion
emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,promo_values)]


y_pos = np.arange(len(promo_values))
bars=ax2.bar(y_pos, promo_values, align='center', color=blue_color, edgecolor='black')
emp_left_bars=ax2.bar(y_pos, emp_left, align='center', color=blue_color, hatch='//',edgecolor='black')

# Add counts on Bars
def autolabel(rects, ax):
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:d}".format(int(rect.get_height())),
                ha='center', va='bottom')
autolabel(bars, ax2)

# Add percentage on Bars
def autolabel_emp(rects, ax):
    index = 0
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:.0f}%".format(int(emp_left_perc[index])),
                ha='center', va='bottom')
        index = index + 1
autolabel_emp(emp_left_bars, ax2)


ax2.set_xticks(y_pos)
ax2.set_xticklabels(["Not-Promotion","Promoted"])
ax2.set_ylabel('Frequency')
ax2.set_title('Employees who left w.r.t Promotion')
ax2.legend([emp_left_bars[0]], ['Employees Left'])

##########################################################
# No. of Projects Bar Chart
proj_count = df['number_project'].value_counts()
proj_indices = proj_count.index.tolist()
proj_values = proj_count.values.tolist()


# Employees left w.r.t No. of Projects
emp_left = []
for p in proj_indices:
    left_emp = len(df.loc[(df['number_project']==p) & (df['left'] == '1')])
    emp_left.append(left_emp)

# Percentage of employees w.r.t No. of Projects
emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,proj_values)]


y_pos = np.arange(len(proj_values))
bars=ax3.bar(y_pos, proj_values, align='center', color=blue_color, edgecolor='black')
emp_left_bars=ax3.bar(y_pos, emp_left, align='center', color=blue_color, hatch='//',edgecolor='black')

# Add counts on Bars
def autolabel(rects, ax):
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:d}".format(int(rect.get_height())),
                ha='center', va='bottom')
autolabel(bars, ax3)

# Add percentage on Bars
def autolabel_emp(rects, ax):
    index = 0
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:.0f}%".format(int(emp_left_perc[index])),
                ha='center', va='bottom')
        index = index + 1
autolabel_emp(emp_left_bars, ax3)
    

ax3.set_xticks(y_pos)
ax3.set_xticklabels(proj_indices)
ax3.set_ylabel('Frequency')
ax3.set_xlabel('No. of Projects')
ax3.set_title('Employees who left w.r.t No. of Projects')
ax3.legend([emp_left_bars[0]], ['Employees Left'])

##########################################################
# Time Spend Bar Chart
spend_count = df['time_spend_company'].value_counts()
spend_indices = spend_count.index.tolist()
spend_values = spend_count.values.tolist()


# Employees left w.r.t Time Spend
emp_left = []
for ts in spend_indices:
    left_emp = len(df.loc[(df['time_spend_company']==ts) & (df['left'] == '1')])
    emp_left.append(left_emp)

# Percentage of employees w.r.t No. of Projects
emp_left_perc = [(ai/bi)*100. for ai,bi in zip(emp_left,spend_values)]


y_pos = np.arange(len(spend_values))
bars=ax4.bar(y_pos, spend_values, align='center', color=blue_color, edgecolor='black')
emp_left_bars=ax4.bar(y_pos, emp_left, align='center', color=blue_color, hatch='//',edgecolor='black')

# Add counts on Bars
def autolabel(rects, ax):
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:d}".format(int(rect.get_height())),
                ha='center', va='bottom')
autolabel(bars, ax4)

# Add percentage on Bars
def autolabel_emp(rects, ax):
    index = 0
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2.,
                rect.get_y() + rect.get_height(),
                "{:.0f}%".format(int(emp_left_perc[index])),
                ha='center', va='bottom')
        index = index + 1
autolabel_emp(emp_left_bars, ax4)
    

ax4.set_xticks(y_pos)
ax4.set_xticklabels(spend_indices)
#ax3.set_xticklabels(["No Work-Accident","Work-Accident"])
ax4.set_ylabel('Frequency')
ax4.set_xlabel('Time Spend in Company')
ax4.set_title('Employees who left w.r.t Time Spend in Comp.')
ax4.legend([emp_left_bars[0]], ['Employees Left'])


plt.show()


# 1. Only 7% of all employees who had work-accident, left the company. Hence 'Work Accident' is not a strong indicator of Employee-Churn.
# 2. Only 5% of all employees who got promoted churned away, as compared to 24% of employees who were not promoted, left the company. Hence promotion is a strong indicator of retaining employees. This is intuitive.
# 3. It is interesting to see that employees with 2, 6 or 7 projects left the company the most. 6 & 7 makes sense i.e. employees quit because of high workload. But it doesn't make sense when No. of Projects = 2. This requires further exploration
# 4. As far as 'Time Spent in Company' is concerned, employees with 6+ years (in company) tend to stay. Employees with 3,4 & 5 years (in company) tend to leave.

# ## Below section shows plots for Avg. Monthly hours & last evaluation

# In[7]:

##########################################################
# Create fiure with 4 subplots
f, ax = plt.subplots(2,2,figsize=(13,13))

(ax1, ax2 , ax3, ax4) = ax.flatten()

##########################################################
# Histogram of Last Eval: I want 20 bins in range (0-1)
ax1.hist(df['last_evaluation'], bins=20, range=(0,1), alpha=0.5, color='b')
ax1.set_title('Histogram: Last-Evaluation')
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Last-Evaluation Scores')



##########################################################
# Histogram of Avg. Monthly Hours: I want 20 bins 
ax2.hist(df['average_montly_hours'], bins=20, alpha=0.5, color='b')
ax2.set_title('Histogram: Avg. Monthly Hours')
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Avg. Monthly Hours')

##########################################################
# Histogram of Last Eval: I want 20 bins in range (0-1)
n, bins, patches = ax3.hist(df['last_evaluation'], bins=20, range=(0,1), alpha=0.5, color='b')
left_in_bins = []
for i in range(len(bins)-1):
    start = bins[i]
    end = bins[i+1]
    
    left_emp = len(df.loc[(df['last_evaluation']>=start) & (df['last_evaluation']<end) & (df['left'] == '1')])
    left_in_bins.append(left_emp)

index = 0
for_legend = None
for p in patches:
    patch = copy.copy(p)
    patch.set_height(left_in_bins[index])
    #patch.set_color(red_color)
    patch.set_hatch('//')
    patch.set_alpha(1.0)
    ax3.add_patch(patch)
    if index==1:
        for_legend = patch
    index = index + 1

ax3.set_title('Histogram: Last-Evaluation with Left Status')
ax3.set_ylabel('Frequency')
ax3.set_xlabel('Last-Evaluation Scores')
ax3.legend([for_legend], ['Employees Left'])
 
##########################################################
# Histogram of Avg. Monthly Hours: I want 20 bins 
n, bins, patches = ax4.hist(df['average_montly_hours'], bins=20, alpha=0.5, color='b')
left_in_bins = []
for i in range(len(bins)-1):
    start = bins[i]
    end = bins[i+1]
    
    left_emp = len(df.loc[(df['average_montly_hours']>=start) & (df['average_montly_hours']<end) & (df['left'] == '1')])
    left_in_bins.append(left_emp)

index = 0
for_legend = None
for p in patches:
    patch = copy.copy(p)
    patch.set_height(left_in_bins[index])
    #patch.set_color(red_color)
    patch.set_hatch('//')
    patch.set_alpha(1.0)
    ax4.add_patch(patch)
    if index==1:
        for_legend = patch
    index = index + 1
ax4.set_title('Histogram: Avg. Monthly Hours with Left Status')
ax4.set_ylabel('Frequency')
ax4.set_xlabel('Avg. Monthly Hours')
ax4.legend([for_legend], ['Employees Left'])

plt.show()


# 1. We see bi-modal distribution in both 'Last-Evaluation' & 'Avg. Monthly Hours' histograms.
# 2. Further, we can see bi-modal distribution of churned-employees within these two histograms.
# 3. Employees with lower 'Last-Evaluation' left the company. But it's a little ambiguous why employees with high 'Last-Evaluation' left. While employees with 0.6-0.8 evaluation score tend to stay.
# 4. Employees with very high working hours tend to leave the company, which makes sense, I could further check what salaries they are getting.

# ## Analysis on why best and most experienced employees are leaving prematurely for this analyse only those employees who: 
# 1. Left the company
# 2. time_spend_company greater than 5
# 3. last_evaluation is greater than 0.72 and less than 1

# In[39]:

experienced_churners = df[(df['time_spend_company']>=5) & (df['left'] == '1') & (df['last_evaluation']> 0.72)&(df['last_evaluation']<=1)]
print(len(experienced_churners[(experienced_churners['number_project'] == 4)|(experienced_churners['number_project'] == 5)]))
print("No. of Employees Left with High experienced {:d}".format(len(experienced_churners)))

f, ax = plt.subplots(3,2,figsize=(12,24))
(ax1, ax2, ax3, ax4, ax5, ax6) = ax.flatten()

#########################################################
#Explore High-Experienced Churners
experienced_churners['salary'].value_counts().plot(kind='bar', ax=ax1, title='Salary')
experienced_churners['department'].value_counts().plot(kind='bar', ax=ax2, title='Department')
experienced_churners['promotion_last_5years'].value_counts().plot(kind='bar', ax=ax3, title='Promotion')
experienced_churners['satisfaction_level'].value_counts().plot(kind='hist', ax=ax4, title='Satisfaction Level.')
experienced_churners['number_project'].value_counts().plot(kind='bar', ax=ax5, title='No. of Projects')
experienced_churners['average_montly_hours'].plot(kind='hist', ax=ax6, title='Avg. Monthly hours')

plt.show()


# # Dissection of Highly-Experinced and Best Employees.
# 1. There were 989 such employees (High-experienced and Best employee Churners)
# 2. There were 600 / 989 working with low salary. Only 1 out of 989 got promoted in last 5 years. Major reason to leave the company i.e. Not got promoted and working with low salary.
# 3. There were 865/989 worked on moderate number of projects. So we can say, they didn't get the number of projects as expected 
# 4. Most of these 989 employees were working very high number of avg. monthly hours and low satisfaction level.
# 
# All of this analysis suggests: Although these employees were highly experienced and best employee for a some reasons, they were not entirely happy.
# It looks like turn-over is function of all these variables:
# No promotion in 5 years -> Leave
# Low Salary+ No-Promotion + High-working-hours -> Leave

# # Model Comparision and Final Model Evaluation

# In[14]:

from sklearn.model_selection import train_test_split

df = pd.read_csv('HR_Data.csv')

# Convert all nominal to numeric.
df['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',
        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)
df['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)
###########################################
# Train & Test Data

df_X = df.copy()
df_y = df_X['left']
del df_X['left']

df_train_X, df_test_X, df_train_Y, df_test_Y = train_test_split(df_X, df_y, test_size = 0.2, random_state = 1234)

print("Train Dataset rows: {} ".format(df_train_X.shape[0]))
print("Test Dataset rows: {} ".format(df_test_X.shape[0]))





# In[21]:

from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, SGDClassifier,LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# list of tuples: the first element is a string, the second is an object
estimators = [('LogisticRegression', LogisticRegression()),('RidgeClassifier', RidgeClassifier()), ('RidgeClassifierCV', RidgeClassifierCV()),              ('RandomForestClassifier', RandomForestClassifier()), ('GradientBoostingClassifier', GradientBoostingClassifier())]

for estimator in estimators:
    scores = cross_val_score(estimator=estimator[1],
                            X=df_train_X,
                            y=df_train_Y,
                            cv=3,
                            scoring='precision', 
                            n_jobs=-1)
    print('CV accuracy scores: %s' % scores)
    print(estimator[0], 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# ### we can say on this data set RandomForestClassifier gives better result than all other algorithms after cross validation 

# In[23]:

###########################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()


param_grid = {
    'n_estimators': [10, 15, 20, 25, 30],
    'max_features': ['auto', 2,3,4,5],
    'min_samples_leaf': [1, 3, 5],
    'criterion': ["gini", "entropy"]
}
CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rfc.fit(df_train_X, df_train_Y)
print(CV_rfc.best_params_)


# ### Below section show the final model using RandomForestClassifier with accuracy 99% on test data set

# In[31]:

#############################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier(n_estimators =30, max_features=3, min_samples_leaf=1, criterion='gini')

rf.fit(df_train_X, df_train_Y)
train_accuracy = rf.score(df_train_X, df_train_Y)
print("Accuracy (Train) {:.04f}".format(train_accuracy))

test_accuracy = rf.score(df_test_X, df_test_Y)
print("Accuracy (Test) {:.04f}".format(test_accuracy))
pred_y = rf.predict(df_test_X)
#prediction_train = rf.predict(df_train_X)
confusion_matrix(df_test_Y, pred_y)
#confusion_matrix(df_train_Y, prediction_train)


# In[27]:

feature_importance = rf.feature_importances_
features_list = df.columns

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

# A threshold below which to drop features from the final data set. Specifically, this number represents
# the percentage of the most important feature's importance value
fi_threshold = 15

# Get the indexes of all features over the importance threshold
important_idx = np.where(feature_importance > fi_threshold)[0]

# Create a list of all the feature names above the importance threshold
important_features = features_list[important_idx]
#print "n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):n", 
#        important_features

# Get the sorted indexes of important features
sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
print("nFeatures sorted by importance (DESC):n" + str(important_features[sorted_idx]))

# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
plt.yticks(pos, important_features[sorted_idx[::-1]])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.draw()
plt.show()


# # Results and Comments
# 1. Satisfaction-Level: Which ever the way it is computed is the strongest predictor of employees who left.
# 2. RandomForest gave a pretty accurate prediction with Training Accuracy of 99.9% and Test Accuracy of 99.0%

# # Confusion Matrix
# array([[2262,    3],
#        [  26,  709]])
# TP = 709, FN = 26
# FP = 3, TN = 2262
# 1. Recall: 96.46% (Percentage of Predicted-Actual-churners(TP) from Total Churners) (TP/TP+FN)
# 2. Precision: 99.57% (Percentage of Actual-churners from Predicted Churners) (TP/TP+FP)
# 3. Test-set Employees Retained = 2265
# 4. Test-set Employees Left = 735
# 5. This Model has more Type II error i.e. more FN (as compared to Type I error), which means some employees who left.
# 6. the company are not predicted as Churners. If this company wants to put more emphasis on retaining, we need to reduce Type II error.
# 7. Although it looks like we are achieving 99% Accuracy (which is very good) but it's useless if Precision & Recall are not evaluated.
