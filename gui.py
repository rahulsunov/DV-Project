from tkinter import *
from PIL import ImageFilter, Image
from tkinter import filedialog, messagebox
import os
import psutil
import time
import subprocess
import cv2
import fnmatch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

main_window = Tk()

def show_entry_fields():
    print("Age: %s\nVeg-NonVeg: %s\nWeight: %s\nHeight: %s\n" % (entry_age.get(), entry_veg.get(), entry_weight.get(), entry_height.get()))

def process_data(input_file):
    data = pd.read_csv(input_file)
    breakfast_data = data['Breakfast']
    breakfast_data_numpy = breakfast_data.to_numpy()

    lunch_data = data['Lunch']
    lunch_data_numpy = lunch_data.to_numpy()

    dinner_data = data['Dinner']
    dinner_data_numpy = dinner_data.to_numpy()

    food_items_data = data['Food_items']
    breakfast_food_separated = []
    lunch_food_separated = []
    dinner_food_separated = []

    breakfast_food_separated_id = []
    lunch_food_separated_id = []
    dinner_food_separated_id = []

    for i in range(len(breakfast_data)):
        if breakfast_data_numpy[i] == 1:
            breakfast_food_separated.append(food_items_data[i])
            breakfast_food_separated_id.append(i)
        if lunch_data_numpy[i] == 1:
            lunch_food_separated.append(food_items_data[i])
            lunch_food_separated_id.append(i)
        if dinner_data_numpy[i] == 1:
            dinner_food_separated.append(food_items_data[i])
            dinner_food_separated_id.append(i)

    lunch_food_separated_id_data = data.iloc[lunch_food_separated_id]
    lunch_food_separated_id_data = lunch_food_separated_id_data.T
    val = list(np.arange(5, 15))
    val_append = [0] + val
    lunch_food_separated_id_data = lunch_food_separated_id_data.iloc[val_append]
    lunch_food_separated_id_data = lunch_food_separated_id_data.T

    breakfast_food_separated_id_data = data.iloc[breakfast_food_separated_id]
    breakfast_food_separated_id_data = breakfast_food_separated_id_data.T
    val = list(np.arange(5, 15))
    val_append = [0] + val
    breakfast_food_separated_id_data = breakfast_food_separated_id_data.iloc[val_append]
    breakfast_food_separated_id_data = breakfast_food_separated_id_data.T

    dinner_food_separated_id_data = data.iloc[dinner_food_separated_id]
    dinner_food_separated_id_data = dinner_food_separated_id_data.T
    val = list(np.arange(5, 15))
    val_append = [0] + val
    dinner_food_separated_id_data = dinner_food_separated_id_data.iloc[val_append]
    dinner_food_separated_id_data = dinner_food_separated_id_data.T

    age = int(entry_age.get())
    veg = float(entry_veg.get())
    weight = float(entry_weight.get())
    height = float(entry_height.get())
    bmi = weight / (height ** 2)
    age_cl = 0

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp + 20)
        for i in test_list:
            if i == age:
                print('Age is between', str(lp), str(lp + 10))
                age_cl = round(lp / 20)

    print("Your body mass index is: ", bmi)
    if bmi < 16:
        print("Severely underweight")
        cl_bmi = 4
    elif 16 <= bmi < 18.5:
        print("Underweight")
        cl_bmi = 3
    elif 18.5 <= bmi < 25:
        print("Healthy")
        cl_bmi = 2
    elif 25 <= bmi < 30:
        print("Overweight")
        cl_bmi = 1
    elif bmi >= 30:
        print("Severely overweight")
        cl_bmi = 0

    val1 = dinner_food_separated_id_data.describe()
    val_tog = val1.T
    print(val_tog.shape)
    print(val_tog)
    dinner_food_separated_id_data = dinner_food_separated_id_data.to_numpy()
    lunch_food_separated_id_data = lunch_food_separated_id_data.to_numpy()
    breakfast_food_separated_id_data = breakfast_food_separated_id_data.to_numpy()
    ti = (cl_bmi + age_cl) / 2

    # K-Means Based Dinner Food
    data_calorie = dinner_food_separated_id_data[1:, 1:len(dinner_food_separated_id_data)]
    X = np.array(data_calorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    dinner_cluster_labels = kmeans.labels_

    data_dinner = pd.DataFrame(dinner_cluster_labels, columns=['Cluster'])
    data_dinner["Cluster"] = data_dinner["Cluster"].astype("category")
    data_dinner = pd.concat([data_dinner, val_tog], axis=1)
    data_dinner.head()

    data_dinner["Cluster"].value_counts()

    # K-Means Based Lunch Food
    data_calorie_lunch = lunch_food_separated_id_data[1:, 1:len(lunch_food_separated_id_data)]
    X_lunch = np.array(data_calorie_lunch)
    kmeans_lunch = KMeans(n_clusters=3, random_state=0).fit(X_lunch)
    lunch_cluster_labels = kmeans_lunch.labels_

    data_lunch = pd.DataFrame(lunch_cluster_labels, columns=['Cluster'])
    data_lunch["Cluster"] = data_lunch["Cluster"].astype("category")
    data_lunch = pd.concat([data_lunch, val_tog], axis=1)
    data_lunch.head()

    data_lunch["Cluster"].value_counts()

    # K-Means Based Breakfast Food
    data_calorie_breakfast = breakfast_food_separated_id_data[1:, 1:len(breakfast_food_separated_id_data)]
    X_breakfast = np.array(data_calorie_breakfast)
    kmeans_breakfast = KMeans(n_clusters=3, random_state=0).fit(X_breakfast)
    breakfast_cluster_labels = kmeans_breakfast.labels_

    data_breakfast = pd.DataFrame(breakfast_cluster_labels, columns=['Cluster'])
    data_breakfast["Cluster"] = data_breakfast["Cluster"].astype("category")
    data_breakfast = pd.concat([data_breakfast, val_tog], axis=1)
    data_breakfast.head()

    data_breakfast["Cluster"].value_counts()

    cl_dinner = round(data_dinner["Cluster"].mean())
    cl_lunch = round(data_lunch["Cluster"].mean())
    cl_breakfast = round(data_breakfast["Cluster"].mean())

    print("Your BMI Cluster is: ", cl_bmi)
    print("Your Age Cluster is: ", age_cl)

    print("Your Dinner Cluster is: ", cl_dinner)
    print("Your Lunch Cluster is: ", cl_lunch)
    print("Your Breakfast Cluster is: ", cl_breakfast)

def browsefunc():
    filename = filedialog.askopenfilename()
    print(filename)
    process_data(filename)

main_window.title("BMI and Diet Recommendation System")
main_window.geometry("500x500")

frame = Frame(main_window)
frame.pack(padx=10, pady=10)

Label(frame, text="Enter your Age:").grid(row=0, column=0)
Label(frame, text="Enter your Veg-NonVeg Preference (0 for Veg, 1 for NonVeg):").grid(row=1, column=0)
Label(frame, text="Enter your Weight (in kg):").grid(row=2, column=0)
Label(frame, text="Enter your Height (in m):").grid(row=3, column=0)

entry_age = Entry(frame)
entry_veg = Entry(frame)
entry_weight = Entry(frame)
entry_height = Entry(frame)

entry_age.grid(row=0, column=1)
entry_veg.grid(row=1, column=1)
entry_weight.grid(row=2, column=1)
entry_height.grid(row=3, column=1)

Button(frame, text='Show', command=show_entry_fields).grid(row=4, column=1, sticky=W, pady=4)
Button(frame, text='Browse', command=browsefunc).grid(row=4, column=2, sticky=W, pady=4)

main_window.mainloop()
