# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, Concatenate
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.optimizers import Adam
#
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# # Load your dataset
# df = pd.read_csv('obesitydata.csv')
#
# # Map string values to numerical values for the 'NObeyesdad' column
# class_mapping = {
#     'Insufficient_Weight': 1,
#     'Normal_Weight': 2,
#     'Obesity_Type_I': 3,
#     'Obesity_Type_II': 4,
#     'Obesity_Type_III': 5,
#     'Overweight_Level_I': 6,
#     'Overweight_Level_II': 7
# }
#
# df['NObeyesdad'] = df['NObeyesdad'].map(class_mapping)
#
# # Separate numerical and categorical columns
# numerical_columns = df.select_dtypes(include=['number']).columns
# categorical_columns = df.select_dtypes(include=['object']).columns
#
# # Drop string columns
# df = df.drop(columns=categorical_columns, errors='ignore')
#
# # Check if 'NObeyesdad' column exists before dropping
# if 'NObeyesdad' in df.columns:
#     # Drop 'NObeyesdad' column from the list of numerical columns
#     numerical_columns = numerical_columns.drop('NObeyesdad', errors='ignore')
#
#     # One-hot encode categorical columns
#     df_encoded = pd.get_dummies(df, drop_first=True)
#
#     # Combine numerical columns with one-hot encoded categorical columns
#     X_encoded = pd.concat([df_encoded[numerical_columns], df_encoded], axis=1)
#
#     # Identify numerical columns
#     numerical_columns = X_encoded.select_dtypes(include=[np.float64, np.int64]).columns
#
#     # Scale numerical columns
#     scaler = MinMaxScaler()
#     X_encoded[numerical_columns] = scaler.fit_transform(X_encoded[numerical_columns])
#
#     # Now, X_encoded contains the scaled values for numerical columns and the original one-hot encoded categorical columns
#     print(X_encoded.head())
# else:
#     print("The 'NObeyesdad' column does not exist in the DataFrame.")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Define the GAN architecture
# def build_generator(latent_dim, n_classes):
#     model = Sequential()
#     model.add(Dense(256, input_dim=latent_dim))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(n_classes, activation='softmax'))
#     return model
#
# def build_discriminator(input_shape, n_classes):
#     model = Sequential()
#     model.add(Dense(1024, input_shape=input_shape, activation='relu'))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(n_classes, activation='softmax'))
#     return model
#
# def build_gan(generator, discriminator):
#     discriminator.trainable = False
#     model = Sequential()
#     model.add(generator)
#     model.add(discriminator)
#     return model
#
# # Prepare the data
# X = df.drop('NObeyesdad', axis=1)
# y = df['NObeyesdad']
#
# # Normalize the numerical features (you might need to handle categorical features differently)
# X = (X - X.min()) / (X.max() - X.min())
#
# # Convert labels to one-hot encoding
# y_onehot = pd.get_dummies(y)
#
# # Set hyperparameters
# latent_dim = 100
# n_classes = y_onehot.shape[1]
#
# # Build and compile the models
# generator = build_generator(latent_dim, n_classes)
# discriminator = build_discriminator(X.shape[1], n_classes)
# discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#
# gan = build_gan(generator, discriminator)
# gan.compile(loss='categorical_crossentropy', optimizer=Adam())
#
# # Training the GAN
# epochs = 10000
# batch_size = 32
#
# for epoch in range(epochs):
#     # Generate random noise as input for the generator
#     noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
#
#     # Generate fake samples with labels
#     gen_samples = generator.predict(noise)
#     labels = np.random.randint(0, n_classes, batch_size)
#     labels_onehot = pd.get_dummies(labels).values
#
#     # Train the discriminator
#     d_loss_real = discriminator.train_on_batch(X.values, pd.get_dummies(y).values)
#     d_loss_fake = discriminator.train_on_batch(gen_samples, labels_onehot)
#     d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#
#     # Train the generator
#     g_loss = gan.train_on_batch(noise, labels_onehot)
#
#     # Print progress
#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}')
#
# # Generate synthetic samples for testing
# num_samples = 100
# noise = np.random.normal(0, 1, size=(num_samples, latent_dim))
# generated_samples = generator.predict(noise)
# generated_labels = np.random.randint(0, n_classes, num_samples)
# generated_labels_onehot = pd.get_dummies(generated_labels).values
#
# # Evaluate the generated samples using the discriminator
# evaluation = discriminator.evaluate(generated_samples, generated_labels_onehot)
# print(f'Generated Samples Evaluation - Loss: {evaluation[0]}, Accuracy: {evaluation[1]}')




import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Function to build the generator model
def build_generator(latent_dim, output_dim):
    input_noise = Input(shape=(latent_dim,))
    x = Dense(256, activation='relu')(input_noise)
    x = Dense(512, activation='relu')(x)
    x = Dense(output_dim, activation='linear')(x)
    generator = Model(inputs=input_noise, outputs=x)
    return generator

# Function to build the discriminator model
def build_discriminator(input_dim):
    input_data = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(input_data)
    x = Dense(256, activation='relu')(x)
    validity = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=input_data, outputs=validity)
    return discriminator

# Function to compile the discriminator model
def compile_discriminator(discriminator, optimizer):
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return discriminator

# Function to compile the combined GAN model
def compile_gan(generator, discriminator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

# Function to train the GAN
def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, latent_dim):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        generated_data = generator.predict(noise)
        real_data = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

        # Label real and generated data
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_data, valid)
        d_loss_fake = discriminator.train_on_batch(generated_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print the progress
        print(f"Epoch {epoch}, [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

# Load your dataset
df = pd.read_csv('obesitydata.csv')

# Select relevant columns for training the GAN (you may need to adjust this based on your requirements)
selected_columns = ['Age', 'Height', 'Weight']

# Normalize data (you may need to use a more sophisticated normalization strategy)
X_train = df[selected_columns].values / 100.0  # Example normalization

# Set hyperparameters
latent_dim = 100
optimizer = tf.keras.optimizers.legacy.Adam(0.0002, 0.5)
epochs = 150
batch_size = 32

# Build and compile the discriminator
discriminator = build_discriminator(X_train.shape[1])
discriminator = compile_discriminator(discriminator, optimizer)

# Build the generator
generator = build_generator(latent_dim, X_train.shape[1])

# Build and compile the GAN model
gan = compile_gan(generator, discriminator, optimizer)

# Train the GAN
train_gan(generator, discriminator, gan, X_train, epochs, batch_size, latent_dim)


import tkinter as tk
from tkinter import simpledialog
import numpy as np

# ... (your previous code)

# Function to get user input using a GUI
def get_user_input():
    user_height = simpledialog.askfloat("Input", "Enter your height in meters:")
    user_weight = simpledialog.askfloat("Input", "Enter your weight in kilograms:")
    user_age = simpledialog.askfloat("Input", "Enter your age:")
    print(f"User Input: Age={user_age}, Height={user_height}, Weight={user_weight}")
    return user_age, user_height, user_weight

# Create a simple GUI to get user input
root = tk.Tk()
root.withdraw()  # Hide the main window

# Get user input
user_age, user_height, user_weight = get_user_input()

# Normalize user input (you may need to use a more sophisticated normalization strategy)
user_data = np.array([[user_age, user_height, user_weight]]) / 100.0  # Example normalization

# Display the normalized user data
print("Normalized User Data:", user_data)

# ... (your remaining code)

# Close the GUI window
# root.destroy()

# # Load your dataset
# df_food = pd.read_csv('input.csv')
#
# # Function to get user input using a GUI
# def get_user_input_food():
#     user_veg_nonveg = simpledialog.askinteger("Input", "Enter 0 for Vegetarian or 1 for Non-Vegetarian:")
#     print(f"User Input - Veg/Non-Veg: {user_veg_nonveg}")
#     return user_veg_nonveg

# Load your dataset
df_food = pd.read_csv('input.csv')

# Function to get user input using a GUI
def get_user_input_food():
    user_veg_nonveg = simpledialog.askinteger("Input", "Enter 0 for Vegetarian or 1 for Non-Vegetarian:")
    print(f"User Input - Veg/Non-Veg: {user_veg_nonveg}")
    return user_veg_nonveg

# Get user input for diet preference
user_veg_nonveg = get_user_input_food()

# Clean the 'VegNovVeg' column by replacing invalid values with 0 (Vegetarian)
df_food['VegNovVeg'] = pd.to_numeric(df_food['VegNovVeg'], errors='coerce').fillna(0).astype(int)

# Select food items based on user preference
selected_category = 'VegNovVeg'

# Filter dataframe based on user input
selected_food_items = df_food[(df_food[selected_category] == user_veg_nonveg) & ((df_food['Breakfast'] == 1) | (df_food['Lunch'] == 1) | (df_food['Dinner'] == 1))]

# Check if there are any rows in the selected_food_items DataFrame
if not selected_food_items.empty:
    # Display 5 random selected food items
    print("\nSelected Food Items:")
    for _, row in selected_food_items.sample(n=5).iterrows():
        print(f"{row['Food_items']} - Veg/Non-Veg: {row['VegNovVeg']}, Breakfast: {row['Breakfast']}, Lunch: {row['Lunch']}, Dinner: {row['Dinner']}")
else:
    print("\nNo food items found based on the given preferences.")

# Print user_veg_nonveg value
print(f"\nUser Veg/Non-Veg Preference: {user_veg_nonveg}")

# Print a few rows of the DataFrame for verification
# print(df_food.head())

# Check if the Tkinter window is still active before attempting to destroy it
if 'root' in locals() and isinstance(root, tk.Tk) and root.winfo_exists():
    root.destroy()



# # Create a Tkinter root window to support the dialog
# root = tk.Tk()
# root.withdraw()  # Hide the main window
#
# # Get user input for diet preference
# user_veg_nonveg = get_user_input_food()
#
# # Select food items based on user preference
# selected_category = 'VegNovVeg'
#
# # Filter dataframe based on user input
# selected_food_items = df_food[(df_food[selected_category] == user_veg_nonveg) & ((df_food['Breakfast'] == 1) | (df_food['Lunch'] == 1) | (df_food['Dinner'] == 1))]
#
# # Check if there are any rows in the selected_food_items DataFrame
# # if not selected_food_items.empty:
# #     # Display random 5 food items
# #     print("\nRandom 5 Food Items:")
# #     random_food_items = selected_food_items.sample(n=5)
# #     for index, row in random_food_items.iterrows():
# #         print(f"{row['Food_items']} - Veg/Non-Veg: {row['VegNovVeg']}, Breakfast: {row['Breakfast']}, Lunch: {row['Lunch']}, Dinner: {row['Dinner']}")
# # else:
# #     print("\nNo food items found based on the given preferences.")
#
# # Select 5 random food items
# random_food_items = df_food.sample(n=5)
#
# # Display the selected food items
# print("\nRandom 5 Food Items:")
# for index, row in random_food_items.iterrows():
#     print(f"{row['Food_items']} - Veg/Non-Veg: {row['VegNovVeg']}, Breakfast: {row['Breakfast']}, Lunch: {row['Lunch']}, Dinner: {row['Dinner']}")
#
# # Print a few rows of the DataFrame for verification
# print("\nDataFrame Head:")
# print(df_food.head())
#
# # Print user_veg_nonveg value
# print(f"\nUser Veg/Non-Veg Preference: {user_veg_nonveg}")
#
# # Display data types of all columns
# print(df_food.dtypes)
#
# # # Print a few rows of the DataFrame for verification
# # print(df_food.head())
#
# # Destroy the Tkinter window
# root.destroy()