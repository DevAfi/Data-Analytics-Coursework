import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def print_section(title):
    print(f"\n{title}")
    print("=" * len(title))


def load_and_prepare_data():
    ratings_df = pd.read_csv("ratings.csv")
    books_df = pd.read_csv("books_new.csv")
    merged_df = pd.merge(ratings_df, books_df, on="bookId", how="inner")

    categorical_cols = merged_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in categorical_cols:
        merged_df[col] = merged_df[col].fillna("Unknown")

    for col in numerical_cols:
        if merged_df[col].isnull().any():
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())

    return ratings_df, books_df, merged_df


def build_book_feature_space(merged_df):
    feature_cols = ["Title", "Author", "Genre", "SubGenre", "Publisher"]
    books_content_df = merged_df[["bookId"] + feature_cols].drop_duplicates(
        subset="bookId"
    ).reset_index(drop=True)

    for col in feature_cols:
        books_content_df[col] = books_content_df[col].fillna("Unknown").astype(str)

    books_content_df["combined_features"] = books_content_df[feature_cols].agg(
        " ".join, axis=1
    )

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(books_content_df["combined_features"])
    cosine_matrix = cosine_similarity(count_matrix, count_matrix)

    # Mapping allows O(1) index lookup for query titles.
    title_to_idx = {title: idx for idx, title in enumerate(books_content_df["Title"])}

    return books_content_df, count_matrix, cosine_matrix, title_to_idx


def vec_space_method(book_title, books_content_df, cosine_matrix, title_to_idx, top_k=10):
    if book_title not in title_to_idx:
        raise ValueError(f"Book '{book_title}' not found in catalogue.")

    target_idx = title_to_idx[book_title]

    # Matrix-vector product (as required): one-hot query vector against similarity matrix.
    query_vector = np.zeros(cosine_matrix.shape[0], dtype=float)
    query_vector[target_idx] = 1.0
    similarity_scores = cosine_matrix @ query_vector

    ranked = np.argsort(similarity_scores)[::-1]
    ranked = [idx for idx in ranked if idx != target_idx][:top_k]

    recs = books_content_df.loc[ranked, ["bookId", "Title"]].copy()
    recs["similarity"] = similarity_scores[ranked]
    recs = recs.sort_values("similarity", ascending=False).reset_index(drop=True)
    return recs


def build_knn_model(count_matrix, n_neighbors=11):
    knn_model = NearestNeighbors(
        metric="cosine",
        algorithm="auto",
        n_neighbors=n_neighbors,
    )
    knn_model.fit(count_matrix)
    return knn_model


def knn_similarity(book_title, books_content_df, count_matrix, knn_model, title_to_idx, top_k=10):
    if book_title not in title_to_idx:
        raise ValueError(f"Book '{book_title}' not found in catalogue.")

    target_idx = title_to_idx[book_title]
    distances, indices = knn_model.kneighbors(
        count_matrix[target_idx],
        n_neighbors=top_k + 1,
    )

    # Drop query book itself.
    neighbor_indices = indices.flatten().tolist()
    neighbor_distances = distances.flatten().tolist()
    pairs = [
        (idx, dist)
        for idx, dist in zip(neighbor_indices, neighbor_distances)
        if idx != target_idx
    ][:top_k]

    recs = books_content_df.loc[[idx for idx, _ in pairs], ["bookId", "Title"]].copy()
    recs["similarity"] = [1.0 - dist for _, dist in pairs]
    recs = recs.sort_values("similarity", ascending=False).reset_index(drop=True)
    return recs


def evaluate_recommenders(
    test_titles,
    vec_method_fn,
    knn_method_fn,
    books_content_df,
):
    catalog_book_ids = set(books_content_df["bookId"].tolist())
    all_book_ids_sorted = sorted(catalog_book_ids)
    id_to_vec_index = {book_id: idx for idx, book_id in enumerate(all_book_ids_sorted)}

    def _collect(method_fn):
        rec_lists = {}
        for title in test_titles:
            recs = method_fn(title)
            rec_lists[title] = recs
        return rec_lists

    def _coverage(rec_lists):
        recommended_ids = set()
        for rec_df in rec_lists.values():
            recommended_ids.update(rec_df["bookId"].tolist())
        return len(recommended_ids) / len(catalog_book_ids)

    def _personalisation(rec_lists):
        # Lecture-aligned: build recommendation vectors, cosine matrix, A from upper triangle.
        user_vectors = []
        for rec_df in rec_lists.values():
            vec = np.zeros(len(all_book_ids_sorted), dtype=float)
            for book_id in rec_df["bookId"]:
                vec[id_to_vec_index[book_id]] = 1.0
            user_vectors.append(vec)

        user_matrix = np.vstack(user_vectors)
        sim_matrix = cosine_similarity(user_matrix)
        upper_idx = np.triu_indices_from(sim_matrix, k=1)
        avg_upper = sim_matrix[upper_idx].mean() if len(upper_idx[0]) > 0 else 0.0
        return 1.0 - avg_upper

    vec_recs = _collect(vec_method_fn)
    knn_recs = _collect(knn_method_fn)

    evaluation_df = pd.DataFrame(
        [
            {
                "method": "VectorSpace",
                "coverage": _coverage(vec_recs),
                "personalisation": _personalisation(vec_recs),
            },
            {
                "method": "KNN",
                "coverage": _coverage(knn_recs),
                "personalisation": _personalisation(knn_recs),
            },
        ]
    )
    return vec_recs, knn_recs, evaluation_df


def train_predict_like_model(merged_df, random_state=42):
    model_df = merged_df.copy()
    model_df["like_label"] = np.where(model_df["rating"] >= 3.6, 1, -1)

    feature_cols = [
        "user_id",
        "bookId",
        "Height",
        "Title",
        "Author",
        "Genre",
        "SubGenre",
        "Publisher",
    ]

    X = model_df[feature_cols]
    y = model_df["like_label"]

    categorical_features = ["Title", "Author", "Genre", "SubGenre", "Publisher"]
    numerical_features = ["user_id", "bookId", "Height"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    knn_classifier = KNeighborsClassifier(n_neighbors=15, weights="distance")
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", knn_classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    return clf, accuracy, report


def predict_like(user_id, book_id, books_df, model):
    book_rows = books_df[books_df["bookId"] == book_id]
    if book_rows.empty:
        raise ValueError(f"bookId {book_id} not found in books_new.csv.")

    book = book_rows.iloc[0]
    input_df = pd.DataFrame(
        [
            {
                "user_id": user_id,
                "bookId": int(book_id),
                "Height": float(book["Height"]),
                "Title": str(book["Title"]),
                "Author": str(book["Author"]) if pd.notna(book["Author"]) else "Unknown",
                "Genre": str(book["Genre"]),
                "SubGenre": str(book["SubGenre"]),
                "Publisher": str(book["Publisher"]) if pd.notna(book["Publisher"]) else "Unknown",
            }
        ]
    )

    prediction = int(model.predict(input_df)[0])
    return prediction


def print_recommendation_block(title, rec_df):
    print(f"\nRecommendations for '{title}':")
    print(rec_df[["Title", "similarity"]].to_string(index=False))


def main():
    np.random.seed(42)
    ratings_df, books_df, merged_df = load_and_prepare_data()
    books_content_df, count_matrix, cosine_matrix, title_to_idx = build_book_feature_space(
        merged_df
    )

    print_section("Task 3.1 - Vector Space Method")
    vec_recs = vec_space_method(
        "Orientalism",
        books_content_df,
        cosine_matrix,
        title_to_idx,
        top_k=10,
    )
    print_recommendation_block("Orientalism", vec_recs)

    print_section("Task 3.2 - KNN Similarity Method")
    knn_model = build_knn_model(count_matrix, n_neighbors=11)
    knn_recs = knn_similarity(
        "Orientalism",
        books_content_df,
        count_matrix,
        knn_model,
        title_to_idx,
        top_k=10,
    )
    print_recommendation_block("Orientalism", knn_recs)

    print_section("Task 3.3 - Recommender Evaluation")
    test_titles = [
        "Fundamentals of Wavelets",
        "Orientalism",
        "How to Think Like Sherlock Holmes",
        "Data Scientists at Work",
    ]

    vec_method_fn = lambda t: vec_space_method(
        t, books_content_df, cosine_matrix, title_to_idx, top_k=10
    )
    knn_method_fn = lambda t: knn_similarity(
        t, books_content_df, count_matrix, knn_model, title_to_idx, top_k=10
    )

    vec_lists, knn_lists, eval_df = evaluate_recommenders(
        test_titles, vec_method_fn, knn_method_fn, books_content_df
    )

    # Evaluation definitions follow the KNN lecture (coverage + personalisation).
    print("Coverage and Personalisation comparison:")
    print(eval_df.to_string(index=False))

    print("\nSample recommendations by method for test users:")
    for title in test_titles:
        print(f"\n[VectorSpace] {title}")
        print(vec_lists[title][["Title", "similarity"]].head(5).to_string(index=False))
        print(f"[KNN] {title}")
        print(knn_lists[title][["Title", "similarity"]].head(5).to_string(index=False))

    print_section("Task 3.4 - predict_like with KNN Classifier")
    clf, accuracy, report = train_predict_like_model(merged_df)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)

    sample_queries = [(314, 5), (439, 1), (1169, 10), (25000, 11)]
    print("Sample predict_like(user_id, book_id) outputs:")
    for user_id, book_id in sample_queries:
        pred = predict_like(user_id, book_id, books_df, clf)
        label = "like" if pred == 1 else "dislike"
        print(f"user_id={user_id}, book_id={book_id} -> {pred} ({label})")


if __name__ == "__main__":
    main()
