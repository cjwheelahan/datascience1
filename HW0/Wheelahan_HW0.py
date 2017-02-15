
# coding: utf-8

# # CS 109A/AC 209A/STAT 121A Data Science: Homework 0
# **Harvard University**<br>
# **Fall 2016**<br>
# **Instructors: W. Pan, P. Protopapas, K. Rader**

# Import libraries

# In[184]:

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from bs4 import BeautifulSoup
import urllib
get_ipython().magic(u'matplotlib inline')


# ## Problem 1: Processing Tabular Data from File
# 
# In this problem, we practice reading csv formatted data and doing some very simple data exploration.

# ### Part (a): Reading CSV Data with Numpy
# 
# Open the file $\mathtt{dataset}$\_$\mathtt{HW0.txt}$, containing birth biometrics as well as maternal data for a number of U.S. births, and inspect the csv formatting of the data. Load the data, without the column headers, into an numpy array. 
# 
# Do some preliminary explorations of the data by printing out the dimensions as well as the first three rows of the array. Finally, for each column, print out the range of the values. 
# 
# <b>Prettify your output</b>, add in some text and formatting to make sure your outputs are readable (e.g. "36x4" is less readable than "array dimensions: 36x4").

# In[185]:

# So start by importing our data
data = np.loadtxt('dataset_HW0.txt', delimiter=',', skiprows=1)

# Let's print the dimensions...
print "The array has the dimensions (rows, columns):"
print data.shape

# and the first three rows...
print "... and the first three rows are:"
print data[:3, :]

# and the range for each of the columns
print "The ranges of each column are as follows:"
for columns in range(3):
    range_list = data[:, columns]
    print "Column", columns, "min is", min(range_list)
    print "Column", columns, "max is", max(range_list)


# ### Part (b): Simple Data Statistics
# 
# Compute the mean birth weight and mean femur length for the entire dataset. Now, we want to split the birth data into three groups based on the mother's age:
# 
# 1. Group I: ages 0-17
# 2. Group II: ages 18-34
# 3. Group III: ages 35-50
# 
# For each maternal age group, compute the mean birth weight and mean femure length. 
# 
# <b>Prettify your output.</b>
# 
# Compare the group means with each other and with the overall mean, what can you conclude?

# In[186]:

# Calculating the mean birth weight and femur length for the whole dataset
print "The mean birth weight is:"
print data[:,1].mean()
print "The mean femur length is:"
print data[:,2].mean()

# Let's create arrays out of our three groups
group1 = data[data[:,2] < 18]
# print group1[:10, :] # A bit of code to check we did this right...

group2 = data[(data[:,2] >= 18) & (data[:,2] < 35)]
# print group2[:10, :] # A bit of code to check we did this right...

group3 = data[data[:,2] >= 35]
# print group3[:10, :] # A bit of code to check we did this right...

print "The mean birth weights for each group are:"
print group1[:,1].mean(), "for group 1,"
print group2[:,1].mean(), "for group 2, and"
print group3[:,1].mean(), "for group 3."

print "The mean femur length for each group are:"
print group1[:,2].mean(), "for group 1,"
print group2[:,2].mean(), "for group 2, and"
print group3[:,2].mean(), "for group 3."

# It looks like as the maternal age increases, the birth weights don't seem to differ in any 
# discernable way, but the femur length increases.


# ### Part (c): Simple Data Visualization
# 
# Visualize the data using a 3-D scatter plot. How does your visual analysis compare with the stats you've computed in Part (b)?

# In[4]:

# Let's create a 3d plot of mother's age vs. femur length vs. birth weight
fig3d = plt.figure()
fig3d = fig3d.add_subplot(111, projection='3d')
fig3d.scatter([data[:,0]], [data[:,1]], [data[:,2]])

# So in this graph, there doesn't seem to be any discernable pattern other than just
# a cloud of data. It's kind of hard to look at and interpret data in this format, though...


# ### Part (d): Simple Data Visualization (Continued)
# 
# Visualize two data attributes at a time,
# 
# 1. maternal age against birth weight
# 2. maternal age against femur length
# 3. birth weight against femur length
# 
# using 2-D scatter plots.
# 
# Compare your visual analysis with your analysis from Part (b) and (c).

# In[5]:

# Plotting mother's age vs birth weight
fig2 = plt.figure()
ma_age_v_bw = fig2.add_subplot(111)
ma_age_v_bw.scatter([data[:,0]], [data[:,2]])

# Plotting maternal age vs. femur length
fig3 = plt.figure()
ma_age_v_fl = fig3.add_subplot(111)
ma_age_v_fl.scatter([data[:,1]], [data[:,2]])

# Plotting birth weight vs. femur length
fig4 = plt.figure()
ma_bw_v_fl = fig4.add_subplot(111)
ma_bw_v_fl.scatter([data[:,0]], [data[:,1]])

# Interesting... the first two plots look very similar. There is a large cluster of births to mothers
# aged 15-17, which (at least visually) tend to plot lower in both birth weight and femur length, but
# from what we saw earlier, we know that visual trends can be deceiving. The third plot, however,
# seems to be much more trendy (shows a discernable trend, trendy... not Pokemon Go, trendy). There
# appears to be a strong positive correlation between femur length and birth weight.


# ### Part (e): More Data Visualization
# 
# Finally, we want to visualize the data by maternal age group. Plot the data again using a 3-D scatter plot, this time, color the points in the plot according to the age group of the mother (e.g. use red, blue, green to represent group I, II and III respectively).
# 
# Compare your visual analysis with your analysis from Part (a) - (c).

# In[6]:

# Let's do the 3d plot again, this time with color for each of the 3 groups
fig3d_v2 = plt.figure()
fig3d_v2 = fig3d_v2.add_subplot(111, projection='3d')
fig3d_v2.scatter([group1[:,0]], [group1[:,1]], [group1[:,2]], c='red')
fig3d_v2.scatter([group2[:,0]], [group2[:,1]], [group2[:,2]], c='green')
fig3d_v2.scatter([group3[:,0]], [group3[:,1]], [group3[:,2]], c='orange')

# Again, it's kind of hard to tell what's going on here, except to see that the data
# is now in 3 bands for the 3 age groups.


# ## Problem 2: Processing Web Data
# 
# In this problem we practice some basic web-scrapping using Beautiful Soup.

# ### Part (a): Opening and Reading Webpages
# 
# Open and load the page (Kafka's The Metamorphosis) at 
# 
# $\mathtt{http://www.gutenberg.org/files/5200/5200-h/5200-h.htm}$
# 
# into a BeautifulSoup object. 
# 
# The object we obtain is a parse tree (a data structure representing all tags and relationship between tags) of the html file. To concretely visualize this object, print out the first 1000 characters of a representation of the parse tree using the $\mathtt{prettify()}$ function.

# In[187]:

# Now for some webscraping

# Let's load a webpage...
page = urllib.urlopen("http://www.gutenberg.org/files/5200/5200-h/5200-h.htm").read()

# plunk it into a Beautiful Soup object,
kafkaesque = BeautifulSoup(page, "lxml")

# and print the first 1000 characters
print kafkaesque.prettify()[0:1000]


# ### Part (b): Exploring the Parsed HTML
# 
# Explore the nested data structure you obtain in Part (a) by printing out the following:
# 
# 1. the content of the head tag
# 2. the text of the head tag
# 3. each child of the head tag
# 2. the text of the title tag
# 3. the text of the preamble (pre) tag
# 4. the text of the first paragraph (p) tag

# In[189]:

# HW0.2b.1
print "The Head contents are:"
print kafkaesque.head.contents

# HW0.2b.2
print "The head text is:"
print kafkaesque.head

# HW0.2b.3
print "Here is the text of each child of the head tag:"
for tag in kafkaesque.head.children:
    print tag.name

# HW0.2b.4
print "The Title is:"
print kafkaesque.title.string

# HW0.2b.5
print "The 'pre' tags contain:"
print kafkaesque.pre.string

# HW0.2b.6
print "The first p tag says:"
print kafkaesque.p.string


# ### Part (c): Extracting Text
# 
# Now we want to extract the text of The Metamorphosis and do some simple analysis. Beautiful Soup provides a way to extract all text from a webpage via the $\mathtt{get}$_$\mathtt{text()}$ function. 
# 
# Print the first and last 5000 characters of the text returned by $\mathtt{get}$_$\mathtt{text()}$. Is this the content of the novela? Where is the content of The Metamorphosis stored in the BeautifulSoup object?

# In[190]:

# Printing the first and last 5000 characters of the webpage
print kafkaesque.get_text()[:5000]
print kafkaesque.get_text()[-5000:]

# Unfortunately, this doesn't look like the actual beginning and end of the text. This is the title of the
# webpage plus the beginning and end of what actually shows on the webpage. The content of the book is
# stored inside the "body" HTML-tag of the page.


# ### Part (d): Extracting Text (Continued)
# 
# Using the $\mathtt{find}$_$\mathtt{all()}$ function, extract the text of all $\mathtt{p}$ tags and concatenate the result into a single string. Print out the first 1000 characters of the string as a sanity check.

# In[191]:

# using find_all to extract the p-tags
p_string = kafkaesque.find_all('p')
text_string = ''

# loop through each iteration of the p tag and append it to 'text_string'
for each in p_string:
    text_string += each.string

# and printing the first 1000 chars
print text_string[:1000]


# ### Part (e): Sentence and Word Count
# 
# Count the number of words in The Metamorphosis. Compute the average word length and plot a histogram of word lengths.
# 
# Count the number of sentences in The Metamorphosis. Compute the average sentence length and plot a histogram of sentence lengths.
# 
# **Hint**: You'll need to pre-process the text in order to obtain the correct word/sentence length and count. 

# In[192]:

# We're gonna create a new list and put each word of the novella as a separate index in that list
print len(text_string.split()), "is the word count." #sanity check
kafka_list= []
for each in text_string.split():
    kafka_list.append(each)

# and we'll print the length of that list, which gives us the word count of the novella
print len(kafka_list), "is still the word count."

# initialize some variables
word_length = []
word_length_tot = 0.00

# then loop through and create a list of word lengths to graph later, and a total word count
for each in kafka_list:
    word_length_tot += len(each)
    word_length.append(len(each))
    
# this will print the average word length
print word_length_tot/len(kafka_list), "is the average word length."

# and we'll plot a histogram of the word length list
word_fig = plt.figure()
word_length_fig = word_fig.add_subplot(111)
word_length_fig.hist(word_length)

# And now to do the same code blocks for sentence lengths
print len(text_string.split('.')), "is the sentence count." #sanity check
sentence_list= []
for each in text_string.split('.'):
    sentence_list.append(each)
print len(sentence_list), "is still the sentence count."
sentence_length = []
sentence_length_tot = 0.00
for each in sentence_list:
    sentence_length_tot += len(each)
    sentence_length.append(len(each))
print sentence_length_tot/len(sentence_list), "is the average sentence length."
# and plot...
sentence_fig = plt.figure()
sentence_length_fig = sentence_fig.add_subplot(111)
print "A histogram of word and sentence lengths looks like this:"
sentence_length_fig.hist(sentence_length)


# ## Problem 3: Data from Simulations
# 
# In this problem we practice generating data by setting up a simulation of a simple phenomenon, a queue. 
# 
# Suppose we're interested in simulating a queue that forms in front of a small Bank of America branch with one teller, where the customers arrive one at a time.
# 
# We want to study the queue length and customer waiting time.

# ### Part (a): Simulating Arrival and Service Time
# 
# Assume that gaps between consecutive arrivals are uniformly distributed over the interval of 1 to 20 minutes (i.e. any two times between 1 minute and 6 minutes are equally likely). 
# 
# Assume that the service times are uniform over the interval of 5 to 15 minutes. 
# 
# Generate the arrival and service times for 100 customers, using the $\mathtt{uniform()}$ function from the $\mathtt{random}$ library.

# In[104]:

# This function will generate a list of 100 arrival intervals
def arr_generator():
    arrival_intervals = []
    arr_interval = 0

    for foo in range(100):
        arr_interval = np.random.randint(1,21)
        arrival_intervals.append(arr_interval)
        
    return arrival_intervals

# This function will generate a list of 100 service intervals
def serv_generator():
    service_intervals = []
    serv_interval = 0
    
    for foo in range(100):
        serv_interval = np.random.randint(5,16)
        service_intervals.append(serv_interval)
    
    return service_intervals

### debug ###
#    print arrival_intervals
#    print service_intervals


# ### Part (b): Simulating the Queue
# 
# Write function that computes the average queue length and the average customer wait time, given the arrival times and the service times.

# In[193]:

def simulation():
    
# Initialize our variables...
    service_intervals = serv_generator()
    arrival_intervals = arr_generator()
    queue = []
    time = 0
    wait_times = [0]
    queue_len = []
    customer_counter = 0
    serv_counter = 0
    arr_time = 0
    waiting = 0.0
    length = 0.0
    last_serv = arrival_intervals[0]
    
# These next few initializations are for debugging
#    service_intervals = [5, 14, 9, 9, 9, 13, 6, 6, 8, 11]
#    arrival_intervals = [5, 5, 17, 10, 17, 10, 1, 8, 7, 10]
#    print arrival_intervals
#    print service_intervals

# For the list we have, we're going to iterate over each of our list items
    while serv_counter < len(service_intervals):

# This block is what we do when we still have list items, and a customer arrives
        if customer_counter < len(arrival_intervals):
            if time == arr_time + arrival_intervals[customer_counter]:
                
# We'll add their arrival time to the time tracker, append their arrival time to the queue, update the average 
# queue length list and iterate the customer number.
                arr_time += arrival_intervals[customer_counter]
                if len(queue) == 0: # just in case the queue is empty, we need to reset the last_served variable
                    last_serv = time
                queue.append(arr_time)
                queue_len.append(len(queue))
                customer_counter += 1
                
# This block is what we do when we still have list items, and a customer's service is complete
# Make sure there's still someone in the queue and check that it's time for a completed service        
        if len(queue) > 0: 
            if time == last_serv + service_intervals[serv_counter]: 
                
                # append the wait time if appropriate, or add 0 if there's nobody in the queue
                if len(queue) == 1:
                    wait_times.append(0)
                else:
                    wait_times.append(time - arr_time)
                
                queue.pop(0) # pop them out of the queue
                last_serv = time # increase the last_serv variable
                serv_counter += 1 # iterate the service counter
                

### Debugging Statements ###        
#        print "At minute", time, "there is/are", len(queue), "people in line."
#        print queue
#        print wait_times
#        print queue_len
#        print serv_counter

        time += 1 #iterate the time counter

# This group of statements is the post-processing to calculate avg wait time and queue length
    wait_times.pop(-1) # The code above will add an extra 0 at the end of the wait times... we just need to nix that
    waiting = sum(wait_times) / len(wait_times)
    length = sum(queue_len) / len(queue_len)

#    print "The average wait time is", waiting, "minutes."
#    print "The average queue length when a new person joins is", length/len(queue_len), "people."
#    print "*** The queue length is inclusive of the person joining and the person currently being served ***"
    return waiting, length

simulation()


# ### Part (c): Average Queue Length and Wait Time
# 
# Run your simulation 500 times and report the mean and std of the average wait time and queue length for 100 customers. What do these statistics mean?
# 
# Explain why is isn't sufficient to run our simulation **once** and report the average wait time/queue length we obtain.

# In[202]:

### The following three functions are from the statistics module of Python 3.4 ###
def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    return sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/n # the population variance
    return pvar**0.5

### Initialize variables ###
function_wait = []
function_queue = []
f1 = 0
f2 = 0
wait = 0
queue = 0

# and just run it 500 times, appending the results into a list
for i in range(500):
    f1, f2 = simulation()
    function_wait.append(f1)
    function_queue.append(f2)
    
# We'll then print the means and stdev of those two lists
print mean(function_wait), "is the mean of the wait times"
print pstdev(function_wait), "is the standard deviation of the wait times"
print mean(function_queue), "is the mean of the queue lengths"
print pstdev(function_queue), "is the standard deviation of the queue lengths"

# The mean is just the average of the list. The standard deviation is essentially the average difference an observation
# can be expected to have from the average. We need to run this simulation a bunch of times to achieve statistical
# significance. We may get 6 for the wait time mean of a single trial, but the mean in this case, over 500 is 2.676.


# ## Problem 4 (Challenge Problem): More Web Scrapping
# 
# In this problem we practice extracting tabular web data. Open and read the webpage at
# 
# $\mathtt{http://www.thisismoney.co.uk/money/news/article-2928285/Economy-tables-GDP-rates-inflation-history-unemployment.html}$
# 
# Extract the Inflation History table and load it into a numpy array.
# 
# Generate a line graph representing the trend of consumer price index vs time (in months).

# In[235]:

from datetime import datetime

# Let's load a webpage...
page = urllib.urlopen("http://www.thisismoney.co.uk/money/news/article-2928285/Economy-tables-GDP-rates_inflation_history-unemployment.html").read()

# plunk it into a Beautiful Soup object,
inflation = BeautifulSoup(page, "lxml")

# Make a list of all our table tags
table_list = inflation.find_all('table')

# Our 4th table tag is the one we need, Let's make a list of the TD tags in there.
td_list = table_list[4].find_all('td')

### Initialize ###
months = []
new_months = []
inf_list = []
num = len(td_list)
index_counter = 0
clean_months = []
clean_inf_list = []

# The next two blocks alternate through our list, and split it into two lists of our months and
# our inflation rates
for i in range(0, num, 2):
    months.append(td_list[i].string)
    
for j in range(1, num, 2):
    inf_list.append(td_list[j].string)

### Debug ###
# print len(months)

# There is some garbage data due to messy tagging on the page... these two blocks strip those rows out.
for count in range(len(months)):
    if months[count] is None:
        index_counter +=1
        
for count in range(len(months)-index_counter):
    if months[count] is None:
        del months[count]
        del inf_list[count]
        
# Still a tiny bit of messy data that the previous blocks didn't strip out.        
del months[2]
del inf_list[2]

# These next two blocks clean the list and turn it into data that can be used in numpy
# **unfortunately I can't seem to get this block to work and I'm running out of time. Everything else should work though...**
########
#for item in months:
#    clean_months.append(int(item))
#for item in inf_list:
#    clean_inf_list.append(float(item))
########

# Create the numpy array and clean it up
########
# array = np.column_stack([clean_months, clean_inf_list])
# array2 = array[1:-3,:] #strip the footers of the table
########

# Graph it
########
# infl_fig = plt.figure()
# inflate = infl_fig.add_subplot(111)
# inflate.plot([array[:,0]], [array[:,1]])
########


# In[ ]:



