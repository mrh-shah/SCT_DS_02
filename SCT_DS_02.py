import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset (assumes 'train.csv' is downloaded locally)
df = pd.read_csv('train.csv')

# Step 1: Basic Info
print("🔍 Dataset Overview:")
print(df.info())
print("\n📊 Summary Statistics:")
print(df.describe())

# Step 2: Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Step 3: Drop Irrelevant Columns
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Step 4: Convert Categorical to Numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 5: Exploratory Data Analysis (EDA)

# Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.legend(['Male', 'Female'])
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='pastel')
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
