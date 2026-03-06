import pandas as pd
import numpy as np

# Part 2.1: Load and combine the datasets

# Load the two CSV files
ratings = pd.read_csv('ratings.csv')
books = pd.read_csv('books_new.csv')

#print(f"   Ratings dataset shape: {ratings.shape}")
#print(f"   Books dataset shape: {books.shape}")

# Combine the contents of both files as a single dataframe

df = pd.merge(ratings, books, on='bookId', how='inner')
#print(f"   Combined dataframe shape: {df.shape}")
#print(f"   Combined dataframe columns: {df.columns.tolist()}")

# Identify categorical and numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
#print(f"   {numerical_features}")

categorical_features = df.select_dtypes(include=['object']).columns.tolist()
#print(f"   {categorical_features}")

# Identify missing values
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

#if len(missing_info) > 0:
#    print("\n   Missing values found:")
#    print(missing_info.to_string())
#else:
#    print("\n   No missing values found in the dataset.")

# Treat missing values
#print("\n5. Treating missing values...")
#print("   Before treatment:")
#print(f"   Total missing values: {df.isnull().sum().sum()}")
#
# For categorical features, fill missing values with 'Unknown'
# For numerical features, we could use mean/median, but let's check if there are any numerical missing values first
#for col in categorical_features:
#    if df[col].isnull().sum() > 0:
#        df[col] = df[col].fillna('Unknown')
#        print(f"   Filled missing values in '{col}' with 'Unknown'")
#
#for col in numerical_features:
#    if df[col].isnull().sum() > 0:
#        # Use median for numerical features (more robust to outliers)
#        median_value = df[col].median()
#        df[col] = df[col].fillna(median_value)
#
#        print(f"   Filled missing values in '{col}' with median: {median_value}")
#print("\n   After treatment:")
#print(f"   Total missing values: {df.isnull().sum().sum()}")

# Show summary statistics
print("\n6. Summary Statistics")
print("=" * 80)

print("\n   Dataset Overview:")
print(f"   Total rows: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

#print("\n   Numerical Features Summary Statistics:")
#print(df[numerical_features].describe().to_string())
#
#print("\n   Categorical Features Summary:")
#for col in categorical_features:
#    print(f"\n   {col}:")
#    print(f"      Unique values: {df[col].nunique()}")
#    print(f"      Most frequent value: {df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'}")
#    print(f"      Frequency: {df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0}")






# TASK 2.2

print("\n7. Task 2.2: Average Ratings and 95% Bootstrap Confidence Intervals")
print("=" * 80)

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


def bootstrap_ci(ratings, bootstrap_num=1000, sample_size=100):
    ratings = np.array(ratings.dropna())
    bootstrap_means = []

    for _ in range(bootstrap_num):
        sample = np.random.choice(ratings, size=sample_size, replace=True)
        bootstrap_means.append(sample.mean())

    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    return lower_bound, upper_bound


ci_lows = []
ci_highs = []

for book_id in top10['bookId']:
    book_ratings = df.loc[df['bookId'] == book_id, 'rating']
    low, high = bootstrap_ci(book_ratings)
    ci_lows.append(low)
    ci_highs.append(high)

top10['ci_low'] = ci_lows
top10['ci_high'] = ci_highs

print("\nTop 10 books by average rating:")
print(top10.to_string(index=False))