# Group 5 — Islamabad Telecom Churn

Predict which customers will leave.  
Target column: `churned` (Yes or No).  
This is a classification task.


## How to run

Open a terminal in this folder, then run:

```text
pip install -r requirements.txt
python main.py
```

Close the chart windows when they open.  
The program prints the answers in the terminal.


## Files

```text
task05_islamabad_churn.csv           customer rows
task05_islamabad_churn_lookup.csv    one row per area
main.py                              all of the work
group05_clean.csv                    cleaned table
group05_dashboard.png                four charts on one page
group05_e1_churn_count.png           chart E1
group05_e2_area_bar.png              chart E2
group05_e3_boxplot.png               chart E3
group05_e4_heatmap.png               chart E4
group05_e5_by_plan.png               chart E5
```


## What main.py does

The file runs from top to bottom. Do not change the order.


### Part A — look at the data

1. Read both CSV files.
2. Print the size, the first rows, the column types, and the missing values.
3. Count exact copy rows.
4. Print the area names. Some names have extra spaces or different capital letters.


### Part B — clean the data

1. Drop exact copy rows. 796 rows become 780.
2. Clean `area`: remove spaces, then capitalise each word.
3. Turn `monthly_bill` from text such as `Rs 1,910` into the number 1910.
4. Turn `last_recharge_on` into a real date.
5. Drop rows where `churned` is missing. 768 rows remain.
6. Drop rows where `tenure_months` is below 0 or above 120. 760 rows remain.
7. Fill empty `data_gb_month` with 12.10. Fill empty `dropped_calls_week` with 3.00.
8. Join the lookup file on `area`. This adds `towers` and `area_type`.
9. Add `flag`. It is 1 when `churned` is Yes, and 0 when `churned` is No.

Final table: 760 rows, 13 columns, 0 missing values.  
Share of rows with `flag` = 1: 0.286.


### Part C — five questions

`flag` mean = share of customers who left.

1. C1. Count customers in each area and the share who left. Highest area: Bara Kahu (0.434).
2. C2. Same share for each area and plan. Highest pair: Bara Kahu + Prepaid (0.435).
3. C3. Same share for each month of `last_recharge_on`. Highest month: 6 (0.384). Lowest month: 7 (0.227).
4. C4. Rural share is 0.387. Urban share is 0.210. Correlation of `towers` and `flag` is -0.202.
5. C5. Prepaid customers with 3 or more complaints. Share who left: 0.571.


### Part D — two simple predictions

Inputs, in this order:

```text
tenure_months, data_gb_month, complaints_6m,
dropped_calls_week, monthly_bill, towers
```

The value to predict is `flag`. The split seed is 2005.

1. Build X (760 rows, 6 columns) and y (760 values).
2. Print the mean and spread of each column. Turn the columns into z-scores. Each z-score column has mean 0.
3. Rank the columns by how strongly they link to `flag`.
   Strongest three: `complaints_6m`, `tenure_months`, `dropped_calls_week`.
4. Split the rows. Train: 604. Test: 156.
   Baseline (always guess the common label): 0.737.
   Straight-line fit: 0.846.
5. The straight line is better by 0.109.


### Part E — charts

1. E1. Count of `churned`, coloured by plan.
2. E2. Bar chart of the share who left, one bar per area.
3. E3. Box plot of `complaints_6m` for customers who stayed and who left.
4. E4. Heatmap of correlations. The numbers are written in the cells.
5. E5. One panel for Prepaid and one panel for Postpaid.
6. E6. Charts E1 to E4 on one page, saved as `group05_dashboard.png`.
7. E7. Outlier check on `complaints_6m`. Values outside the fences: 0. Nothing is removed.


### Part F — save

The cleaned table is saved as `group05_clean.csv`.


## Report still to write

Write at most two pages. Include:

1. Problems found in Part A, and how many rows each cleaning step removed.
2. A line that the check values match the task page.
3. Three findings from Part C, each with its number.
4. Two charts, each with one sentence on what it shows.
5. The three strongest features, plus 0.737 and 0.846.
6. One short paragraph: the final model predicts `flag`, uses the six features, and must beat 0.846.
