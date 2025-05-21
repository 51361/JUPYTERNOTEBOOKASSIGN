# JUPYTERNOTEBOOKASSIGN



# Data Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Data Loading
print("1. Loading data...")
try:
    # Load sample dataset (replace with your actual data source)
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)
    print("Titanic dataset loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Data Exploration
print("\n2. Exploring the data...")

# Basic information
print("\nData shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types and missing values:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Basic Data Analysis
print("\n3. Performing basic analysis...")

# Survival rate analysis
survival_rate = df['survived'].mean()
print(f"\nOverall survival rate: {survival_rate:.2%}")

# Survival by class
class_survival = df.groupby('class')['survived'].mean()
print("\nSurvival rate by passenger class:")
print(class_survival)

# Age distribution of survivors vs non-survivors
age_analysis = df.groupby('survived')['age'].describe()
print("\nAge distribution by survival status:")
print(age_analysis)

# 4. Data Visualizations
print("\n4. Creating visualizations...")

# Figure 1: Survival count
plt.figure()
sns.countplot(x='survived', data=df)
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.show()

# Figure 2: Survival by passenger class
plt.figure()
sns.barplot(x='class', y='survived', data=df, ci=None)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Figure 3: Age distribution by survival
plt.figure()
sns.histplot(data=df, x='age', hue='survived', bins=30, kde=True, element='step')
plt.title('Age Distribution by Survival Status')
plt.show()

# Figure 4: Fare distribution
plt.figure()
sns.boxplot(x='class', y='fare', data=df)
plt.title('Fare Distribution by Passenger Class')
plt.yscale('log')  # Using log scale due to extreme outliers
plt.show()

# Figure 5: Correlation heatmap
plt.figure()
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 5. Findings and Observations
print("\n5. Key Findings and Observations:")

print("\n1. Survival Rate:")
print(f"- Overall survival rate was {survival_rate:.2%}")
print("- First class passengers had significantly higher survival rates (62.9%) compared to third class (24.2%)")

print("\n2. Age Distribution:")
print("- Average age of survivors was slightly lower (28.34 years) than non-survivors (30.63 years)")
print("- The youngest passengers (children) had higher survival rates")

print("\n3. Passenger Class Impact:")
print("- First class passengers paid significantly higher fares (as expected)")
print("- Class appears to be strongly correlated with survival (higher class = better chance)")

print("\n4. Missing Data:")
print("- Age has 177 missing values (about 20% of data)")
print("- Cabin information is mostly missing (77% missing)")

print("\n5. Interesting Correlations:")
print("- Survival shows moderate positive correlation with fare (0.26)")
print("- Negative correlation between passenger class and age (-0.37)")

print("\nRecommendations for Further Analysis:")
print("- Investigate the relationship between age, sex, and survival (women and children first policy)")
print("- Explore embarked port as a potential factor in survival")
print("- Handle missing age data (imputation or removal)")

