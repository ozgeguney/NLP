import numpy as np
import pickle
import pandas as pd

data = pd.read_csv('/Users/ozgeguney/PycharmProjects/pythonProject/data/capitals.txt', delimiter=' ')
data.columns = ['city1','country1','city2','country2']
print(data.head(5))
word_embeddings = pickle.load( open( "/Users/ozgeguney/PycharmProjects/pythonProject/data/word_embeddings_subset.p", "rb" ) )
king = word_embeddings['king']
queen = word_embeddings['queen']
print(type(word_embeddings.keys()))
def cosine_similarity(A,B):
    dot = np.dot(A,B)
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    cos = dot/(normA*normB)
    return cos
print(cosine_similarity(king,queen))

def euclidean(A, B):
    d = np.sqrt(np.sum((A - B) ** 2))
    return d
print(euclidean(king, queen))

def get_country(city1, country1, city2, embeddings, cosine_similarity=cosine_similarity):
    group= set((city1,country1,city2))
    city1_emb = embeddings[city1]
    country1_emb = embeddings[country1]
    city2_emb = embeddings[city2]
    vec = country1_emb - city1_emb + city2_emb
    similarity = -1
    country=''
    for word in embeddings.keys():
        if word not in group:
            word_emb= embeddings[word]
            current_similarity = cosine_similarity(vec, word_emb)
            if current_similarity> similarity:
                similarity = current_similarity
                country = (word,similarity)
    return country

print(get_country('Athens', 'Greece', 'Berlin', word_embeddings))

def get_accuracy(word_embeddings, data, get_country=get_country):
    num_correct = 0
    for i, row in data.iterrows():
        city1 = row.city1
        country1 = row.country1
        city2 = row.city2
        country2 = row.country2
        predicted_country,_ = get_country(city1,country1,city2,word_embeddings)
        if country2 == predicted_country:
            num_correct +=1
    m = len(data)
    accuracy= num_correct / m
    return accuracy

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")

def get_vectors(embeddings, words):
    m = len(words)
    X = np.zeros((1, 300))
    for word in words:
        english = word
        eng_emb = embeddings[english]
        X = np.row_stack((X, eng_emb))
    X = X[1:, :]
    return X

def compute_PCA(X, n_components=2):
    X_demeaned = X - np.mean(X.T, axis=1)
    covariance_matrix = np.cov(X_demeaned.T, rowvar=True)
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
    idx_sorted = np.argsort(eigen_vals)
    idx_sorted_decreasing = idx_sorted[::-1]
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]
    eigen_vecs_subset = eigen_vecs_sorted[:, :n_components]
    X_reduced = np.dot(eigen_vecs_subset.T, X_demeaned.T).T
    return X_reduced

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# given a list of words and the embeddings, it returns a matrix with all the embeddings
X = get_vectors(word_embeddings, words)

import matplotlib.pyplot as plt
result = compute_PCA(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

plt.show()