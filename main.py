# Group 5 — customers leaving a telecom company in Islamabad.
# Target: churned (Yes or No). This is a classification task.
# Run this file from the D:\project folder:  python main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# =============================================================================
# Part A. Load the files and look at them before changing anything.
# =============================================================================

# Main table: one row is one customer. Lookup: one row is one area.
data = pd.read_csv("task05_islamabad_churn.csv")
lookup = pd.read_csv("task05_islamabad_churn_lookup.csv")

print("Shape of the customer file (rows, columns):", data.shape)
print("First rows of the customer file:")
print(data.head())
print()
print("Shape of the lookup file (rows, columns):", lookup.shape)
print("Lookup file:")
print(lookup)
print()

# Column types, a number summary, and how many holes are in each column.
print("Column types:")
data.info()
print()
print("Number summary:")
print(data.describe())
print()
print("Missing values in each column:")
print(data.isna().sum())
print()

# Exact copy rows, and the area names we will use to join the lookup.
print("Exact copy rows:", data.duplicated().sum())
print("Area names in the customer file:")
print(data["area"].value_counts())
print("Area names in the lookup file:")
print(lookup["area"].value_counts())
print()

# Problems we can already see. The program fixes these in Part B.
impossible_tenure = (data["tenure_months"] < 0) | (data["tenure_months"] > 120)
print("Problems before cleaning:")
print("- Exact copy rows:", int(data.duplicated().sum()))
print("- Missing churned (the target):", int(data["churned"].isna().sum()))
print("- Impossible tenure_months (below 0 or above 120):", int(impossible_tenure.sum()))
print("- Missing data_gb_month:", int(data["data_gb_month"].isna().sum()))
print("- Missing dropped_calls_week:", int(data["dropped_calls_week"].isna().sum()))
print("- monthly_bill is text such as 'Rs 1,910', not a number")
print("- area has extra spaces and different capital letters, so it will not match the lookup")
print()


# =============================================================================
# Part B. Clean in the order given on the task page.
# A different order gives different check numbers.
# =============================================================================

print("Rows in the raw file", len(data))

# Step 1. Drop rows that are exact copies of an earlier row. Keep the first copy.
clean = data.drop_duplicates().copy()
print("Rows after step 1 (exact copies dropped)", len(clean))

# Step 2. Remove spaces at both ends of area, then capitalise each word.
# Example: "shahzad town" becomes "Shahzad Town", "Saddar  " becomes "Saddar".
clean["area"] = clean["area"].str.strip().str.title()
print("Rows after step 2 (area cleaned)", len(clean))

# Step 3. Turn monthly_bill from text into a number.
# "Rs 1,910" -> remove Rs -> remove the comma -> 1910.
clean["monthly_bill"] = (
    clean["monthly_bill"]
    .str.replace("Rs", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
clean["monthly_bill"] = pd.to_numeric(clean["monthly_bill"])
print("Rows after step 3 (monthly_bill is a number)", len(clean))

# Step 4. Turn last_recharge_on from text into a real date.
clean["last_recharge_on"] = pd.to_datetime(clean["last_recharge_on"])
print("Rows after step 4 (dates parsed)", len(clean))

# Step 5. Remove rows where the target, churned, is missing.
clean = clean.loc[~clean["churned"].isna()].copy()
print("Rows after step 5 (missing target removed)", len(clean))

# Step 6. Remove impossible tenure. Keep rows where tenure is missing.
# A missing tenure compared with 0 or 120 is False, so those rows stay.
bad_tenure = (clean["tenure_months"] < 0) | (clean["tenure_months"] > 120)
clean = clean.loc[~bad_tenure].copy()
print("Rows after step 6 (impossible tenure_months removed)", len(clean))

# Step 7. Fill holes in two columns with the middle value (the median).
# Take the median now, after the bad rows are gone and before the holes are filled.
med_data = clean["data_gb_month"].median()
med_drops = clean["dropped_calls_week"].median()
print("Median used to fill data_gb_month", f"{med_data:.2f}")
print("Median used to fill dropped_calls_week", f"{med_drops:.2f}")
clean["data_gb_month"] = clean["data_gb_month"].fillna(med_data)
clean["dropped_calls_week"] = clean["dropped_calls_week"].fillna(med_drops)
print("Rows after step 7 (missing numbers filled)", len(clean))

# Step 8. Add towers and area_type from the lookup. Keep every customer row.
clean = clean.merge(lookup, on="area", how="left")
print("Rows after step 8 (lookup merged)", len(clean))

# Step 9. flag is 1 when the customer left, and 0 otherwise.
# Part D predicts this 0/1 column.
clean["flag"] = (clean["churned"] == "Yes").astype(int)

print("Rows and columns of the final table", clean.shape[0], "and", clean.shape[1])
print("Missing values left in the final table", int(clean.isna().sum().sum()))
print("Share of rows with flag equal to 1", round(clean["flag"].mean(), 3))
print()


# =============================================================================
# Part C. Five questions. Each answer is a number plus one sentence.
# =============================================================================

# C1. How many customers are in each area, and what share left?
# The share is the mean of flag, because flag is 1 for Yes and 0 for No.
print("C1. Customers and churn share by area")
c1 = clean.groupby("area").agg(
    customers=("customer_id", "size"),
    churn_share=("flag", "mean"),
)
print(c1.round(3))
best_area = c1["churn_share"].idxmax()
print(
    "Highest area:",
    best_area,
    "with churn share",
    round(c1.loc[best_area, "churn_share"], 3),
)
print("Sentence: " + best_area + " has the highest share of customers who left.")
print()

# C2. Churn share for every pair of area and plan.
print("C2. Churn share by area and plan")
c2 = clean.pivot_table(values="flag", index="area", columns="plan", aggfunc="mean")
print(c2.round(3))

best_value = -1
best_area_name = ""
best_plan_name = ""
for area_name in c2.index:
    for plan_name in c2.columns:
        value = c2.loc[area_name, plan_name]
        if value > best_value:
            best_value = value
            best_area_name = area_name
            best_plan_name = plan_name
print("Highest pair:", best_area_name, "+", best_plan_name, round(best_value, 3))
print(
    "Sentence: "
    + best_area_name
    + " customers on the "
    + best_plan_name
    + " plan have the highest share who left."
)
print()

# C3. Churn share by the month of the last recharge.
print("C3. Churn share by month of last_recharge_on")
c3 = clean.groupby(clean["last_recharge_on"].dt.month)["flag"].mean()
print(c3.round(3))
print("Highest month:", int(c3.idxmax()), "share", round(c3.max(), 3))
print("Lowest month:", int(c3.idxmin()), "share", round(c3.min(), 3))
print(
    "Sentence: month "
    + str(int(c3.idxmax()))
    + " has the highest churn share and month "
    + str(int(c3.idxmin()))
    + " has the lowest."
)
print()

# C4. Rural versus urban, then the link between tower count and leaving.
print("C4. Churn share by area type, and correlation of towers with flag")
c4 = clean.groupby("area_type")["flag"].mean()
print(c4.round(3))
towers_corr = np.corrcoef(clean["towers"], clean["flag"])[0, 1]
print("Correlation between towers and flag:", round(towers_corr, 3))
print(
    "Sentence: Rural churn share is "
    + str(round(c4["Rural"], 3))
    + " and Urban churn share is "
    + str(round(c4["Urban"], 3))
    + ". The correlation between towers and flag is "
    + str(round(towers_corr, 3))
    + "."
)
print()

# C5. Prepaid customers who complained 3 or more times.
print("C5. Prepaid customers with 3 or more complaints")
prepaid_many = (clean["plan"] == "Prepaid") & (clean["complaints_6m"] >= 3)
c5_share = clean.loc[prepaid_many, "flag"].mean()
print("Customers in this group:", int(prepaid_many.sum()))
print("Share who left:", round(c5_share, 3))
print(
    "Sentence: among prepaid customers with three or more complaints, "
    + str(round(c5_share, 3))
    + " left."
)
print()


# =============================================================================
# Part D. Build number arrays, then two simple predictions.
# =============================================================================

# These six columns are the inputs. flag is what we predict.
features = [
    "tenure_months",
    "data_gb_month",
    "complaints_6m",
    "dropped_calls_week",
    "monthly_bill",
    "towers",
]
units = {
    "tenure_months": "months with the company",
    "data_gb_month": "GB per month",
    "complaints_6m": "complaints in 6 months",
    "dropped_calls_week": "dropped calls per week",
    "monthly_bill": "PKR per month",
    "towers": "towers in the area",
}

# .values turns the columns into a plain number grid.
X = clean[features].values
y = clean["flag"].values
print("D1. Shape of X (rows, features):", X.shape)
print("D1. Shape of y (rows):", y.shape)
print()

# D2. Mean and spread of each feature, then z-scores.
# A z-score says how far a value is from the column average.
# NumPy std divides by n. Do not use pandas std here.
print("D2. Mean and standard deviation of each feature")
for i, name in enumerate(features):
    print(" ", name, "mean", round(X[:, i].mean(), 3), "std", round(X[:, i].std(), 3))

Z = (X - X.mean(axis=0)) / X.std(axis=0)
print("Every z-score column has mean 0:", np.allclose(Z.mean(axis=0), 0))
print()

# D3. Which features move together with flag? Ignore the plus or minus sign
# when we rank them. A negative correlation still counts as a strong link.
print("D3. Correlation of each feature with flag")
corr_rows = []
for i, name in enumerate(features):
    r = np.corrcoef(X[:, i], y)[0, 1]
    corr_rows.append((name, r))

corr_table = pd.DataFrame(corr_rows, columns=["feature", "correlation"])
corr_table["size"] = corr_table["correlation"].abs()
corr_table = corr_table.sort_values("size", ascending=False)
print(corr_table[["feature", "correlation"]].round(3).to_string(index=False))

strongest = corr_table.iloc[0]["feature"]
second = corr_table.iloc[1]["feature"]
third = corr_table.iloc[2]["feature"]
print("Strongest three:", strongest + ",", second + ",", third)
print()

# D4. Split rows with seed 2005. About 80 percent are for training.
# The seed keeps the same split for every person who runs this file.
# Do not sort the table before this split.
rng = np.random.default_rng(2005)
in_train = rng.random(len(clean)) < 0.8
X_train, X_test = X[in_train], X[~in_train]
y_train, y_test = y[in_train], y[~in_train]
print("Part D: train rows and test rows (seed 2005)", len(y_train), "and", len(y_test))

# Baseline: ignore every feature. Always predict the more common training label.
# If more than half of training customers left, that label is 1. Otherwise it is 0.
if y_train.mean() > 0.5:
    majority = 1
else:
    majority = 0
baseline_accuracy = np.mean(y_test == majority)
print("Part D: baseline accuracy (always the more common label)", round(baseline_accuracy, 3))

# Standardise using the training rows only, then fit a straight line with least squares.
# A score of 0.5 or more is called 1 (left). Below 0.5 is called 0 (stayed).
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
Z_train = (X_train - mean) / std
Z_test = (X_test - mean) / std

A_train = np.hstack([np.ones((len(Z_train), 1)), Z_train])
A_test = np.hstack([np.ones((len(Z_test), 1)), Z_test])
w = np.linalg.solve(A_train.T @ A_train, A_train.T @ y_train)
predicted = np.where(A_test @ w >= 0.5, 1, 0)
linear_accuracy = np.mean(predicted == y_test)
print("Part D: least-squares accuracy", round(linear_accuracy, 3))

# D5. One sentence on what the straight line gained.
gain = round(linear_accuracy - baseline_accuracy, 3)
print(
    "D5. The straight-line fit gained "
    + str(gain)
    + " accuracy over the baseline on the same test rows."
)
print()


# =============================================================================
# Part E. Charts. Each file is saved in this folder.
# Close the chart windows to let the program finish.
# =============================================================================

# E1. How many customers left, coloured by Prepaid or Postpaid.
plt.figure(figsize=(8, 5))
sns.countplot(data=clean, x="churned", hue="plan")
plt.title("Customers by churn, split by plan")
plt.xlabel("Churned in the next three months")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig("group05_e1_churn_count.png", dpi=150)

# E2. Share who left in each area.
plt.figure(figsize=(8, 5))
area_share = clean.groupby("area")["flag"].mean().sort_values(ascending=False)
area_share.plot(kind="bar", color="steelblue")
plt.title("Share of customers who left, by area")
plt.xlabel("Area")
plt.ylabel("Share with flag = 1")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("group05_e2_area_bar.png", dpi=150)

# E3. Box plot of the strongest feature for customers who stayed and who left.
plt.figure(figsize=(8, 5))
sns.boxplot(data=clean, x="churned", y=strongest)
plt.title("Strongest feature for customers who stayed and who left")
plt.xlabel("Churned in the next three months")
plt.ylabel(strongest + " (" + units[strongest] + ")")
plt.tight_layout()
plt.savefig("group05_e3_boxplot.png", dpi=150)

# E4. Heatmap of correlations between the features and flag.
plt.figure(figsize=(8, 6))
corr = clean[features + ["flag"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation of features and flag")
plt.tight_layout()
plt.savefig("group05_e4_heatmap.png", dpi=150)

# E5. One panel for Prepaid and one panel for Postpaid.
g = sns.catplot(data=clean, x="churned", col="plan", kind="count")
g.fig.suptitle("Churn counts in each plan")
g.set_axis_labels("Churned in the next three months", "Number of customers")
g.fig.subplots_adjust(top=0.85)
g.savefig("group05_e5_by_plan.png", dpi=150)

# E6. The first four charts on one 2 by 2 dashboard.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=clean, x="churned", hue="plan", ax=axes[0, 0])
axes[0, 0].set_title("Customers by churn, split by plan")
axes[0, 0].set_xlabel("Churned in the next three months")
axes[0, 0].set_ylabel("Number of customers")

area_share.plot(kind="bar", color="steelblue", ax=axes[0, 1])
axes[0, 1].set_title("Share of customers who left, by area")
axes[0, 1].set_xlabel("Area")
axes[0, 1].set_ylabel("Share with flag = 1")
axes[0, 1].tick_params(axis="x", rotation=30)

sns.boxplot(data=clean, x="churned", y=strongest, ax=axes[1, 0])
axes[1, 0].set_title("Strongest feature by churn label")
axes[1, 0].set_xlabel("Churned in the next three months")
axes[1, 0].set_ylabel(strongest + " (" + units[strongest] + ")")

sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axes[1, 1])
axes[1, 1].set_title("Correlation of features and flag")

fig.tight_layout()
fig.savefig("group05_dashboard.png", dpi=150)

# E7. Outlier check on the strongest feature.
# The fences sit 1.5 times the interquartile range beyond the quartiles.
s = clean[strongest]
q1 = s.quantile(0.25)
q3 = s.quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5 * iqr
high = q3 + 1.5 * iqr
n_out = int(((s < low) | (s > high)).sum())
print("E7. Outlier check for", strongest)
print("Q1", round(q1, 3), "Q3", round(q3, 3), "IQR", round(iqr, 3))
print("Lower fence", round(low, 3), "Upper fence", round(high, 3))
print("Values outside the fences:", n_out)
if n_out == 0:
    print("There are no outliers on this feature, so nothing is removed.")
else:
    print(
        "These "
        + str(n_out)
        + " values are unusual but still possible for a real customer."
    )
    print("I would keep them. The task only tells us to remove impossible tenure.")
print()


# =============================================================================
# Part F. Save the cleaned table. The final project starts from this file.
# =============================================================================

clean.to_csv("group05_clean.csv", index=False)
print("Saved group05_clean.csv")
print("Saved group05_dashboard.png and the five chart files.")
print(
    "Final project: predict flag (1 = churned Yes) for a new customer. "
    "Use the six features above. Do not use customer_id. "
    "The model must beat the least-squares accuracy of "
    + str(round(linear_accuracy, 3))
    + "."
)

plt.show()
