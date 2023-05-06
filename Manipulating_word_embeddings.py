import pandas as pd # Library for Dataframes
import numpy as np # Library for math functions
import pickle # Python object serialization library. Not secure

word_embeddings = pickle.load( open( "/Users/ozgeguney/PycharmProjects/pythonProject/data/word_embeddings_subset.p", "rb" ) )
print(len(word_embeddings))

countryVector = word_embeddings['country'] # Get the vector representation for the word 'country'
print(type(countryVector))
print(countryVector)

def vec(w):
    return word_embeddings[w]


import matplotlib.pyplot as plt

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']

bag2d = np.array([vec(word) for word in words])  # Convert each word to its vector representation

fig, ax = plt.subplots(figsize=(10, 10))  # Create custom size image

col1 = 3  # Select the column for the x axis
col2 = 2  # Select the column for the y axis

# Print an arrow for each word
for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width=0.005, head_length=0.005, fc='r', ec='r', width=1e-5)

ax.scatter(bag2d[:, col1], bag2d[:, col2]);  # Plot a dot for each word

# Add the word label over each dot in the scatter plot
for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))

#plt.show()

words = ['sad', 'happy', 'town', 'village']

bag2d = np.array([vec(word) for word in words])  # Convert each word to its vector representation

fig, ax = plt.subplots(figsize=(10, 10))  # Create custom size image

col1 = 3  # Select the column for the x axe
col2 = 2  # Select the column for the y axe

# Print an arrow for each word
for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width=0.0005, head_length=0.0005, fc='r', ec='r', width=1e-5)

# print the vector difference between village and town
village = vec('village')
town = vec('town')
diff = town - village
ax.arrow(village[col1], village[col2], diff[col1], diff[col2], fc='b', ec='b', width=1e-5)

# print the vector difference between village and town
sad = vec('sad')
happy = vec('happy')
diff = happy - sad
ax.arrow(sad[col1], sad[col2], diff[col1], diff[col2], fc='b', ec='b', width=1e-5)

ax.scatter(bag2d[:, col1], bag2d[:, col2]);  # Plot a dot for each word

# Add the word label over each dot in the scatter plot
for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))

#plt.show()

print(np.linalg.norm(vec('town'))) # Print the norm of the word town
print(np.linalg.norm(vec('sad'))) # Print the norm of the word sad

capital = vec('France') - vec('Paris')
country = vec('Madrid') + capital

print(country[0:5])

diff = country - vec('Spain')
print(diff[0:10])

keys = word_embeddings.keys()
data = []
for key in keys:
    data.append(word_embeddings[key])

embedding = pd.DataFrame(data=data, index=keys)
# Define a function to find the closest word to a vector:
def find_closest_word(v, k = 1):
    # Calculate the vector difference from each word to the input vector
    diff = embedding.values - v
    # Get the squared L2 norm of each difference vector.
    # It means the squared euclidean distance from each word to the input vector
    delta = np.sum(diff * diff, axis=1)
    # Find the index of the minimun distance in the array
    i = np.argmin(delta)
    # Return the row name for this item
    return embedding.iloc[i].name

print(embedding.head(10))

print(find_closest_word(country))

print(find_closest_word(vec('Italy') - vec('Rome') + vec('Madrid')))

print(find_closest_word(vec('Berlin') + capital))
print(find_closest_word(vec('Beijing') + capital))

print(find_closest_word(vec('Lisbon') + capital))

doc = "Spain petroleum city king"
vdoc = [vec(x) for x in doc.split(" ")]
doc2vec = np.sum(vdoc, axis = 0)
doc2vec

print(find_closest_word(doc2vec))