# Aravjit Sachdeva
# UTA ID: 1001383194
import numpy as np
import pandas as pd
import math
import sys
import random
import collections

def pdf(x, mean, std):
    std = std**2
    return (np.exp(-np.power(x - mean, 2.) / (2 * np.power(std, 2.)))) / (std * np.power(2 * math.pi, 1/2))

def main():

	training_data = pd.read_csv(sys.argv[1], delim_whitespace=True)
	number_of_cols = len(training_data.columns)
	array = []
	for i in range(number_of_cols):
	    array.append(i+1)

	training_data = pd.read_csv(sys.argv[1], delim_whitespace=True, names = array )
	test_data = pd.read_csv(sys.argv[2], delim_whitespace=True, names = array)
	#test_data.drop(9, axis=1, inplace = True)




	classes = np.array(training_data.iloc[:, -1].unique())
	classes.sort()
	number_of_classes = len(classes)
	class_number=1
	df_list = []
	num=0
	x=1
	class_probabilites = {}
	list_of_attributes = []
	list_of_pvalues = []
	gaussian = []
	attribute = []
	classified =[]
	temp_array = []
	p_clazz = np.zeros((len(classes)))
	for u in training_data.iloc[:,-1]:
	    # get index of current clazz value
	    index_clazz = np.where(classes == u)

	    # increment 1 the count of the intersection of buying x clazz pair.
	    p_clazz[index_clazz] += 1

	# let's normalize the possibilities
	p_clazz = p_clazz / np.sum(p_clazz, axis=0, keepdims=True)

	# we create a pandas dataframe to visualize the table more familiar
	df_p_clazz = pd.DataFrame(p_clazz, classes)

	# possibilities of class values
	df_p_clazz.sort_index(inplace=True)

	for i in range(number_of_classes):
	    df_list.append(training_data[training_data[array[-1]]==classes[i]])
	    df_list[i].drop(array[-1], axis =1, inplace = True)
	    class_number = class_number+1



	for i in range(len(df_list)):
	    
	    for j in range(len(df_list[i].columns)):
	        attribute.append(classes[i])
	        attribute.append(j+1)
	        col_name = df_list[i].columns[j]
	        mean = df_list[i][col_name].mean()
	        attribute.append(mean)
	        std = df_list[i][col_name].std()
	        if std>=0.01:
	            attribute.append(std)
	        else:
	            attribute.append(0.01)
	        list_of_attributes.append(attribute)
	        attribute = []
	        
	count = 1
	counter= 0
	counter2 = 0
	number_tied = 0
	for i in range(len(list_of_attributes)):
	        print("Class %d , Attribute %d,  mean %.2f, std %.2f" % (list_of_attributes[i][0], list_of_attributes[i][1], list_of_attributes[i][2], list_of_attributes[i][3]) )

	list_of_attributes.append([-1,-1,-1,-1])


	    
	for j in range(len(list_of_attributes)):
	    if num <len(classes):
	        if list_of_attributes[j][0]==classes[num] and list_of_attributes[j][0] != -1:

	            gaussian = pdf(test_data[count], list_of_attributes[j][2], list_of_attributes[j][3])

	            x = x*gaussian
	            
	            count = count+1
	            if list_of_attributes[j+1][0] == classes[num]+1 or list_of_attributes[j+1][0]==-1:
	                #print(x*p_clazz[num])
	                list_of_pvalues.append(x*p_clazz[num])
	                
	                num = num+1
	                count= 1

	                x=1


	p_arr = np.array(list_of_pvalues)

	p = p_arr / np.sum(p_arr, axis=0)

	classified = []
	accuracy = []

	for j in range(len(p[0])):
	    temp_array = p_arr[:,j]
	    index = np.where(temp_array == max(temp_array))
	    if len(index[0])>0:
	    	idx1 = index[0][random.choice(index[0])]
	    else:
	    	idx1 = index[0][0]
	    
	    
	    temp_array = []

	    

	    

	#classification = np.array([k + 1 for idx in p.T for k in range(len(idx)) if idx[k] == np.max(idx)])
	test_labels = np.array(test_data[[array[-1]]]).T[0]
	print('Classification phase: ')


	for i in range(len(classified)):
	    if classified[i]==test_labels[i]:
	        counter = collections.Counter(p_arr[:, i])
	        max_val_freq = counter[max(p_arr[:, i])]
	        accuracy.append(1/max_val_freq)
	        
	    else:
	        accuracy.append(0)
	#for i in range(len(classified)):
	#    if classified[i]==test_labels[i]:
	#        counter=counter+1
	        
	for i in range(len(test_labels)):
	    print("ID= %5d, predicted= %3d, probability= %.4f, true= %3d, accuracy = %4.2f" % (i+1, classified[i], max(p_arr[:,i]), test_labels[i], accuracy[i] ))
	print("classifier accuracy: %6.4f" % (sum(accuracy)/len(accuracy)))        

      


main()