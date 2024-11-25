#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:8885/predict'

customer_id = 'xyz-123'
customer = {'age': 39,
 'workclass': ' State-gov',
 'fnlwgt': 77516,
 'education': ' Bachelors',
 'education-num': 13,
 'marital-status': ' Never-married',
 'occupation': ' Adm-clerical',
 'relationship': ' Not-in-family',
 'race': ' White',
 'sex': ' Male',
 'capital-gain': 2174,
 'capital-loss': 0,
 'hours-per-week': 40,
 'native-country': ' United-States'
 }

response = requests.post(url, json=customer).json()
print(response)

if response['income'] == True:
    print(f"The annual income for the {customer_id} is greater than $50K.")
else:
    print(f"The annual income for the {customer_id} is below or equal to $50K.")