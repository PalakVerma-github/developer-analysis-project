# =========================================================
# STACK OVERFLOW DEVELOPER SURVEY ANALYSIS PROJECT
# =========================================================

# -------------------------------
# IMPORTING LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# FOLDER TO SAVE CHARTS
# -------------------------------
if not os.path.exists("charts"):
    os.makedirs("charts")

# -------------------------------
# PASTEL THEME 
# -------------------------------
sns.set_theme(style="whitegrid", context="talk")

pastel_colors = [
    "#f8c8dc",  # pastel pink
    "#fde2a7",  # pastel yellow
    "#cdeac0",  # pastel green
    "#b5ead7",  # pastel mint
    "#c7ceea",  # pastel lavender
    "#ffdac1",  # pastel peach
    "#a2d2ff",  # pastel blue
    "#d7bde2"   # pastel purple
]

sns.set_palette(pastel_colors)

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/python project/survey_results_public.csv")

# -------------------------------
# BASIC CHECK
# -------------------------------
print("First 5 rows:\n")
print(df.head())

print("\nDataset shape:", df.shape)

# -------------------------------
# SELECT IMPORTANT COLUMNS
# -------------------------------
df = df[
    [
        "Age",
        "EdLevel",
        "Employment",
        "WorkExp",
        "LanguageHaveWorkedWith",
        "ConvertedCompYearly",
        "JobSat"
    ]
]

print("\nSelected columns:\n", df.columns)
print("\nShape after selecting columns:", df.shape)

# -------------------------------
# DATA CLEANING
# -------------------------------
# Converting important columns into numeric format for analysis
# Removing missing salary values and filtering extreme outliers
# Filling missing categorical and numeric values for cleaner analysis
# Convert numeric columns safely
df["ConvertedCompYearly"] = pd.to_numeric(df["ConvertedCompYearly"], errors="coerce")
df["WorkExp"] = pd.to_numeric(df["WorkExp"], errors="coerce")
df["JobSat"] = pd.to_numeric(df["JobSat"], errors="coerce")

# Remove rows where salary is missing
df = df[df["ConvertedCompYearly"].notna()]

# Remove extreme salary outliers
df = df[df["ConvertedCompYearly"] < 1000000]

# Fill missing categorical values with mode
df["EdLevel"] = df["EdLevel"].fillna(df["EdLevel"].mode()[0])
df["Employment"] = df["Employment"].fillna(df["Employment"].mode()[0])

# Fill numeric missing values with median
df["WorkExp"] = df["WorkExp"].fillna(df["WorkExp"].median())
df["JobSat"] = df["JobSat"].fillna(df["JobSat"].median())

# Drop rows where language is missing
df = df.dropna(subset=["LanguageHaveWorkedWith"])

# Remove duplicates
# Duplicate rows are removed to improve data quality
df = df.drop_duplicates()

print("\nMissing values after cleaning:\n")
print(df.isnull().sum())

print("\nShape after cleaning:", df.shape)

# -------------------------------
# CLEANING EDUCATION LABELS
# -------------------------------
df["EdLevel"] = df["EdLevel"].replace({
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master's",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "High School",
    "Some college/university study without earning a degree": "Some College",
    "Associate degree (A.A., A.S., etc.)": "Associate",
    "Primary/elementary school": "Primary",
    "Other (please specify):": "Other"
})
# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["ExperienceLevel"] = pd.cut(
    df["WorkExp"],
    bins=[0, 2, 5, 10, 20, 50],
    labels=["Beginner", "Junior", "Mid-Level", "Senior", "Expert"],
    include_lowest=True
)

print("\nExperience level distribution:\n")
print(df["ExperienceLevel"].value_counts())

# -------------------------------
# CREATE SEPARATE LANGUAGE DATAFRAME
# -------------------------------
language_df = df.copy()

language_df = language_df.assign(
    LanguageHaveWorkedWith=language_df["LanguageHaveWorkedWith"].str.split(";")
).explode("LanguageHaveWorkedWith")

language_df = language_df.reset_index(drop=True)

print("\nLanguage dataframe shape:", language_df.shape)
print("\nTop 10 languages:\n", language_df["LanguageHaveWorkedWith"].value_counts().head(10))

# =========================================================
# PLOT 1: TOP 10 PROGRAMMING LANGUAGES
# =========================================================
plt.figure(figsize=(12,7))

# Get top 10 languages
top_languages = language_df["LanguageHaveWorkedWith"].value_counts().head(10)
top_languages.plot(
    kind="bar",
    color=pastel_colors[:len(top_languages)],  
    edgecolor="black"
)

plt.title("Top 10 Programming Languages", fontweight="bold")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig("charts/top_languages.png", dpi=300)
plt.show()

# =========================================================
# PLOT 2: SALARY BY EXPERIENCE LEVEL
# =========================================================
plt.figure(figsize=(12,7))

sns.boxplot(
    x="ExperienceLevel",
    y="ConvertedCompYearly",
    data=df,
    palette=pastel_colors
)

plt.title("Salary Distribution by Experience Level", fontweight="bold")

plt.tight_layout()
plt.savefig("charts/salary_experience.png", dpi=300)
plt.show()


# =========================================================
# PLOT 3: TOP PROGRAMMING LANGUAGES BY EXPERIENCE LEVEL
# =========================================================

# Step 1: Get top 5 languages
top_langs = language_df["LanguageHaveWorkedWith"].value_counts().head(5).index

# Step 2: Filter dataset for only top languages
filtered_lang = language_df[
    language_df["LanguageHaveWorkedWith"].isin(top_langs)
]

# Step 3: Plot
plt.figure(figsize=(12,7))

sns.countplot(
    data=filtered_lang,
    x="LanguageHaveWorkedWith",
    hue="ExperienceLevel",
    palette=pastel_colors,   # ensures pastel colors are applied
    edgecolor="black"
)

plt.title("Top Programming Languages by Experience Level", fontweight="bold")
plt.xlabel("Programming Language")
plt.ylabel("Number of Developers")

plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig("charts/languages_by_experience.png", dpi=300)
plt.show()

# =========================================================
# PLOT 4: SALARY BY EDUCATION LEVEL
# =========================================================
plt.figure(figsize=(12,7))

sns.boxplot(
    x="EdLevel",
    y="ConvertedCompYearly",
    data=df,
    palette=pastel_colors
)

plt.title("Salary by Education Level", fontweight="bold")

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("charts/salary_education.png", dpi=300)
plt.show()

# =========================================================
# PLOT 5: JOB SATISFACTION VS SALARY
# =========================================================
plt.figure(figsize=(12,7))

sns.boxplot(
    x="JobSat",
    y="ConvertedCompYearly",
    data=df,
    palette=pastel_colors
)

plt.title("Salary vs Job Satisfaction", fontweight="bold")

plt.tight_layout()
plt.savefig("charts/jobSat_salary.png", dpi=300)
plt.show()

# =========================================================
# PLOT 6: CORRELATION HEATMAP
# =========================================================

# Create numeric dataframe
numeric_df = df[["WorkExp", "ConvertedCompYearly", "JobSat"]]

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="Pastel1",   # pastel theme
    linewidths=1,
    linecolor="white"
)

plt.title("Correlation Between Work Experience, Salary and Job Satisfaction", fontweight="bold")

plt.tight_layout()
plt.savefig("charts/correlation.png", dpi=300)
plt.show()
# =========================================================
# PLOT 7: MEDIAN SALARY BY TOP PROGRAMMING LANGUAGES
# =========================================================

# Step 1: Get top 5 languages
top_langs = language_df["LanguageHaveWorkedWith"].value_counts().head(5).index

# Step 2: Filter only top languages
salary_lang = language_df[
    language_df["LanguageHaveWorkedWith"].isin(top_langs)
]

# Step 3: Calculate median salary by language
median_salary = salary_lang.groupby("LanguageHaveWorkedWith")["ConvertedCompYearly"].median().sort_values()

# Step 4: Plot
plt.figure(figsize=(10,6))

median_salary.plot(
    kind="barh",
    color=pastel_colors[:len(median_salary)],
    edgecolor="black"
)

plt.title("Median Salary by Top Programming Languages", fontweight="bold")
plt.xlabel("Median Salary")
plt.ylabel("Programming Language")

plt.tight_layout()
plt.savefig("charts/median_salary.png", dpi=300)
plt.show()



# =========================================================
# PROJECT OBJECTIVES and CONCLUSION
# =========================================================

print("\n================= PROJECT OBJECTIVES =================")

print("1. To identify the most popular programming languages used by developers based on the Stack Overflow survey dataset.")

print("2. To analyze the relationship between developers’ work experience and their annual salary.")

print("3. To compare salary distribution across different education levels and understand its impact on earnings.")

print("4. To examine the relationship between job satisfaction and salary among developers.")

print("5. To explore correlations between key variables such as work experience, salary, and job satisfaction to uncover meaningful patterns.")

print("\n================= CONCLUSION =================")

print("This project analyzed the Stack Overflow Developer Survey dataset to uncover meaningful insights about programming trends, salaries, education levels, and job satisfaction among developers.")

print("Through data cleaning, feature engineering, and transformation of multi-valued columns, the dataset was prepared for effective analysis.")

print("Advanced visualizations such as boxplots, heatmaps, and grouped comparisons helped in understanding relationships between variables like experience, salary, and satisfaction.")

print("Overall, the project demonstrates how Python can be used for real-world data analysis, enabling data-driven insights and professional decision-making.")