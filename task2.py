import pandas as pd
import numpy as np


def print_section(title):
    print(f"\n{title}")
    print("=" * len(title))


def bootstrap_ci(ratings, n_boot=1000, sample_size=100):
    ratings = np.array(ratings.dropna())
    boot_means = []

    for _ in range(n_boot):
        sample = np.random.choice(ratings, size=sample_size, replace=True)
        boot_means.append(sample.mean())

    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)
    return lower, upper


np.random.seed(42)

# Part 2.1: load, combine, inspect and clean the data
ratings = pd.read_csv('ratings.csv')
books = pd.read_csv('books_new.csv')
df = pd.merge(ratings, books, on='bookId', how='inner')

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

missing_info = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(
    'Missing Count',
    ascending=False
)

for col in categorical_features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna('Unknown')

for col in numerical_features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print_section("Task 2.1")
print("Numerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nMissing values before treatment:")
if missing_info.empty:
    print("No missing values found.")
else:
    print(missing_info.to_string())

print("\nSummary statistics:")
print(df[numerical_features].describe().to_string())

# Task 2.2: average rating and bootstrap confidence intervals
avg_rating = (
    df.groupby(['bookId', 'Title'])['rating']
    .mean()
    .reset_index(name='average_rating')
)

top10 = (
    avg_rating.sort_values('average_rating', ascending=False)
    .head(10)
    .copy()
)

ci_lows = []
ci_highs = []

for book_id in top10['bookId']:
    book_ratings = df.loc[df['bookId'] == book_id, 'rating']
    low, high = bootstrap_ci(book_ratings)
    ci_lows.append(low)
    ci_highs.append(high)

top10['ci_low'] = ci_lows
top10['ci_high'] = ci_highs

print_section("Task 2.2")
print("Top 10 books by average rating with 95% bootstrap confidence intervals:")
print(top10.to_string(index=False))