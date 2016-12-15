#!/bin/bash

python new_main.py data/useful_interactions.csv data/target_users.csv data/item_profile.csv --header 0 --recommender user_knn --params similarity=cosine,k=25,shrinkage=0 --prediction_file output/predictions.csv --rec_length 5

python CBF/cbf_user_main.py data/user_profile.csv --header 0 --recommender cbf_users --params similarity=cosine,k=25,shrinkage=10 --rec_length 5  --attr jobroles

python CBF/cbf_prediction.py data/item_profile.csv data/target_users.csv data/interactions.csv output/new_cbf.csv


python CBF/cbf_user_main.py data/item_profile.csv --header 0 --key id --recommender cbf_items --params similarity=cosine,k=25,shrinkage=10 --rec_length 5 --attr title

python cf_users_main.py data/interactions.csv --recommender item_knn --params similarity=cosine,k=25,shrinkage=0 --rec_length 5


python cf_prediction.py data/item_profile.csv data/target_users.csv data/interactions.csv output/collaborative_filtering2.csv