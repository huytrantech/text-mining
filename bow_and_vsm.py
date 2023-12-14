from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Dữ liệu mẫu
documents = [
    "natural language processing great",
    "Natural language processing subfield artificial intelligence.",
    "Machine learning "
]

# Sử dụng CountVectorizer để tạo Bag of Words (BoW)
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(documents)

# Sử dụng TfidfVectorizer để tạo Vector Space Model (VSM)
tfidf_vectorizer = TfidfVectorizer()
vsm_matrix = tfidf_vectorizer.fit_transform(documents)

# Hiển thị kết quả
print("Bag of Words (BoW) Matrix:")
print(bow_matrix.toarray())
print("Feature names (words):", count_vectorizer.get_feature_names_out())
print("\nVector Space Model (VSM) Matrix:")
print(vsm_matrix.toarray())
print("Feature names (words):", tfidf_vectorizer.get_feature_names_out())
