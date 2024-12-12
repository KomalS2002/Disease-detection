import csv
import pickle
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from itertools import combinations
from time import time
from nltk.tokenize import RegexpTokenizer
from collections import OrderedDict 

# Load English stop words and initialize a lemmatizer for text preprocessing
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Load the symptoms data from a pickle file
with open('final_sysmptoms.pickle', 'rb') as handle:
    dis_symp = pickle.load(handle)
    
t0 = time()  # Start time measurement

# Initialize sets and dictionaries for storing processed data
total_symptoms = set()  # Stores all unique symptoms
diseases_symptoms_cleaned = OrderedDict()  # Key: disease, Value: List of cleaned symptoms

# Iterate over all diseases and preprocess their symptoms
for key in sorted(dis_symp.keys()):
    value = dis_symp[key]
    # Remove annotations in square brackets and convert to lowercase
    list_sym = re.sub(r"\[\S+\]", "", value).lower().split(',')
    
    # Remove empty or unwanted entries (e.g., 'none')
    temp_sym = list_sym
    list_sym = []
    for sym in temp_sym:
        if len(sym.strip()) > 0:
            list_sym.append(sym.strip())
    if "none" in list_sym: 
        list_sym.remove("none")
    if len(list_sym) == 0:
        continue

    temp = []  # Temporary list for cleaned symptoms
    for sym in list_sym:
        # Replace special characters and tokenize the text
        sym = sym.replace('-', ' ').replace("'", '').replace('(', '').replace(')', '')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym) \
                        if word not in stop_words and not word[0].isdigit()])
        total_symptoms.add(sym)  # Add symptom to the set of unique symptoms
        temp.append(sym)
    
    # Add the cleaned symptoms list to the dictionary for the current disease
    diseases_symptoms_cleaned[key] = temp

# Convert the set of unique symptoms to a sorted list and add the label column
total_symptoms = list(total_symptoms)
total_symptoms.sort()
total_symptoms = ['label_dis'] + total_symptoms

print(len(diseases_symptoms_cleaned))  # Print the number of processed diseases

t1 = time()  # Record the preprocessing end time
print(t1 - t0)  # Print the preprocessing duration

# Initialize two dataframes: one for normal data and one for combinations
df_comb = pd.DataFrame(columns=total_symptoms)
df_norm = pd.DataFrame(columns=total_symptoms)

# Process each disease and its symptoms to populate the dataframes
for key, values in diseases_symptoms_cleaned.items():
    key = str.encode(key).decode('utf-8')  # Ensure the key is UTF-8 encoded
    
    # Create a dictionary for the normal dataset row and populate it
    row_norm = dict({x: 0 for x in total_symptoms})
    for sym in values:
        row_norm[sym] = 1
    row_norm['label_dis'] = key
    df_norm = df_norm.append(pd.Series(row_norm), ignore_index=True)
         
    # Create dictionaries for all combinations of symptoms for the combination dataset
    for comb in range(1, len(values) + 1):
        for subset in combinations(values, comb):
            row_comb = dict({x: 0 for x in total_symptoms})
            for sym in list(subset):
                row_comb[sym] = 1
            row_comb['label_dis'] = key
            df_comb = df_comb.append(pd.Series(row_comb), ignore_index=True)

print(df_comb.shape)  # Print the shape of the combination dataset
print(df_norm.shape)  # Print the shape of the normal dataset

# Export the datasets to CSV files for further use
df_comb.to_csv("dis_sym_dataset_comb.csv", index=None)
df_norm.to_csv("dis_sym_dataset_norm.csv", index=None)

t2 = time()  # Record the end time of the entire process
print(t2 - t1)  # Print the dataset generation duration

# Export the cleaned disease-symptoms dictionary to a text file for better visibility
with open('dis_symp_dict.txt', 'w') as f:
    for key, value in diseases_symptoms_cleaned.items():
        print([key] + value, file=f)
