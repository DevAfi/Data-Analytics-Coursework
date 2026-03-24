import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

# Task 2.3: add rating_count, analyse relationship, suggest significance threshold
book_rating_stats = (
    df.groupby(['bookId', 'Title'])['rating']
    .agg(average_rating='mean', rating_count='count')
    .reset_index()
)

rating_count_unique = book_rating_stats['rating_count'].nunique()
if rating_count_unique <= 1:
    corr_value = np.nan
else:
    corr_value = book_rating_stats['average_rating'].corr(
        book_rating_stats['rating_count']
    )

print_section("Task 2.3")
print("Book rating statistics (average_rating + rating_count):")
print(book_rating_stats.head(15).to_string(index=False))

print("\nCorrelation between average_rating and rating_count:")
if np.isnan(corr_value):
    print("Undefined (rating_count is constant in this merged dataset).")
else:
    print(f"{corr_value:.4f}")

# Plot helps show that low-count books can have extreme average ratings.
plt.figure(figsize=(10, 6))
plt.scatter(
    book_rating_stats['rating_count'],
    book_rating_stats['average_rating'],
    alpha=0.6
)
plt.xscale('log')
plt.xlabel('Rating Count (log scale)')
plt.ylabel('Average Rating')
plt.title('Average Rating vs Number of Ratings per Book')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('task2_3_rating_vs_count.png', dpi=200)
plt.close()

# Data-driven threshold when variation exists; otherwise report observed constant.
threshold = max(10, int(np.ceil(book_rating_stats['rating_count'].quantile(0.25))))
significant_ratings = book_rating_stats[book_rating_stats['rating_count'] >= threshold]

print(f"\nSuggested significance threshold for rating_count: {threshold}")
print(
    f"Books with at least {threshold} ratings: "
    f"{len(significant_ratings)} / {len(book_rating_stats)}"
)
if rating_count_unique <= 1:
    print(
        "Commentary: In this merged dataset (books_new.csv joined with ratings.csv), "
        "all books have the same rating_count (100). Therefore, no relationship between "
        "average_rating and rating_count can be inferred here, and any threshold "
        "suggestion is trivial for this subset."
    )
else:
    print(
        "Commentary: Books with very low rating_count are more likely to show "
        "unstable (often extreme) average ratings. A minimum rating_count threshold "
        "reduces noise and makes average ratings more reliable."
    )

# Task 2.4: binary like/dislike transform and content-based recommendations
df['rating_binary'] = np.where(df['rating'] >= 3.6, 1, -1)

print_section("Task 2.4")
print("Binary rating transform using threshold 3.6:")
print(df['rating_binary'].value_counts().rename_axis('class').to_string())

selected_features = ['Title', 'Author', 'Genre', 'SubGenre', 'Publisher']
books_content_df = df[['bookId'] + selected_features].drop_duplicates(
    subset='bookId'
).reset_index(drop=True).copy()

for feature in selected_features:
    books_content_df[feature] = books_content_df[feature].fillna('Unknown').astype(str)

books_content_df['combined_features'] = books_content_df[selected_features].agg(
    ' '.join, axis=1
)

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(books_content_df['combined_features'])
cosine_sim_matrix = cosine_similarity(count_matrix, count_matrix)

print("\nCosine similarity matrix shape:")
print(cosine_sim_matrix.shape)

target_title = 'Orientalism'
target_matches = books_content_df.index[books_content_df['Title'] == target_title].tolist()

if not target_matches:
    print(f"\nBook '{target_title}' not found in the current merged dataset.")
else:
    target_idx = target_matches[0]

    # Matrix-vector product: similarity scores against all books from one query book.
    query_vector = np.zeros(cosine_sim_matrix.shape[0], dtype=float)
    query_vector[target_idx] = 1.0
    similarity_scores = cosine_sim_matrix @ query_vector

    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_indices = [idx for idx in ranked_indices if idx != target_idx][:10]

    recommendations = books_content_df.loc[ranked_indices, ['Title']].copy()
    recommendations['similarity'] = similarity_scores[ranked_indices]
    recommendations = recommendations.sort_values('similarity', ascending=False)

    print(f"\nTop 10 recommendations for a user who liked '{target_title}':")
    print(recommendations.to_string(index=False))