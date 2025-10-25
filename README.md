# 🧠 Adult Income Classification (UCI Census Dataset)

This project analyzes and classifies census data from the **UCI Adult dataset** to predict whether an individual's income exceeds $50K/year based on demographic attributes.

## 📋 Project Overview

The UCI Adult dataset contains census information such as age, education, occupation, hours worked, and more.  
The task is to classify each record as earning **">50K"** or **"<=50K"** annually.

This project includes:
1. **Data Exploration & Visualization**
2. **Data Preprocessing & Cleaning**
3. **Model Training & Evaluation**
4. **Performance Comparison Across Models**

---

## 📊 Dataset Details

**Source:** [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

| Property | Description |
|-----------|-------------|
| **File used** | `adult.data` |
| **Number of instances** | ~32,000 |
| **Number of attributes** | 14 (plus target label) |
| **Target variable** | `income` → whether income is `<=50K` or `>50K` |
| **Type of problem** | Binary Classification |

### Main Features
- `age` (continuous)
- `workclass` (categorical)
- `education` (categorical)
- `marital-status` (categorical)
- `occupation` (categorical)
- `relationship` (categorical)
- `race` (categorical)
- `sex` (categorical)
- `hours-per-week` (continuous)
- `native-country` (categorical)


---

### 1️⃣ Clone the repository
```bash
git clone https://github.com/AsianTaquito/census_data_miniProject.git
cd census_data_miniProject
