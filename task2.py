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
    if ratings.size == 0:
        raise ValueError("Cannot bootstrap confidence interval from empty ratings.")
    boot_means = []
    effective_sample_size = min(sample_size, ratings.size)

    for _ in range(n_boot):
        sample = np.random.choice(ratings, size=effective_sample_size, replace=True)
        boot_means.append(sample.mean())

    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)
    return lower, upper


np.random.seed(42)

# Task 2.1: load, merge, then handle missing values
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

# Task 2.2: average rating per book + bootstrap CI for overall mean
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

overall_ci_low, overall_ci_high = bootstrap_ci(df['rating'], n_boot=1000, sample_size=100)

print_section("Task 2.2")
print("Top 10 books by average rating:")
print(top10.to_string(index=False))
print(
    "\n95% bootstrap CI for the overall mean rating "
    f"(1000 samples of size 100): [{overall_ci_low:.4f}, {overall_ci_high:.4f}]"
)
# Task 2.3: add rating_count, check rating vs count, and suggest a sensible cutoff
# Count ratings straight from ratings.csv so this reflects raw user activity
rating_count_df = (
    ratings.groupby('bookId')['rating']
    .size()
    .reset_index(name='rating_count')
)
avg_rating_df = (
    ratings.groupby('bookId')['rating']
    .mean()
    .reset_index(name='average_rating')
)
book_rating_stats = books[['bookId', 'Title']].merge(
    avg_rating_df,
    on='bookId',
    how='left'
).merge(
    rating_count_df,
    on='bookId',
    how='left'
)
book_rating_stats['rating_count'] = book_rating_stats['rating_count'].fillna(0).astype(int)

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
    print("Undefined (rating_count is constant in the available ratings data).")
else:
    print(f"{corr_value:.4f}")

# Quick check for whether low counts give noisy averages
plot_df = book_rating_stats[book_rating_stats['rating_count'] > 0].copy()

# Pick a threshold from the data when possible.
threshold = max(10, int(np.ceil(plot_df['rating_count'].quantile(0.25))))
significant_ratings = plot_df[plot_df['rating_count'] >= threshold]

print(f"\nSuggested significance threshold for rating_count: {threshold}")
print(
    f"Books with at least {threshold} ratings: "
    f"{len(significant_ratings)} / {len(book_rating_stats)}"
)
if rating_count_unique <= 1:
    print(
        "Commentary: all books have the same rating_count here, "
        "so this correlation is not meaningful."
    )
else:
    print(
        "Commentary: books with very low rating_count can look unusually high or low "
        "just by chance. Using a minimum count gives more stable averages."
    )

# Task 2.4: turn ratings into like/dislike and build content-based recs
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

    # Matrix-vector product: one query vector against all book vectors.
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