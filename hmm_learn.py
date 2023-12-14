import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

# Example training data for POS tagging
# Each tuple represents (word, POS)
training_data = [
    ('Will', 'DET'), ('cat', 'NOUN'), ('is', 'VERB'), ('on', 'PREP'),
    ('the', 'DET'), ('mat', 'NOUN'), ('.', 'PUNCT')
]

# Extract observations (words) and labels (POS)
observations, labels = zip(*training_data)

# Convert labels to numerical values using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels).reshape(-1, 1)

# Convert observations to numerical values (you may use word embeddings in a real scenario)
encoded_observations = np.array(range(len(observations))).reshape(-1, 1)

# Create and train the HMM model
model = hmm.MultinomialHMM(n_components=len(label_encoder.classes_))
model.fit(encoded_observations)

# Predict the hidden states (POS tags) for a new sequence
new_observations = np.array(range(len(observations))).reshape(-1, 1)
predicted_hidden_states = model.predict(new_observations)

# Map numerical labels back to POS tags
predicted_labels = label_encoder.inverse_transform(predicted_hidden_states)

# Display the predicted POS tags
for word, pos in zip(observations, predicted_labels):
    print(f"{word}: {pos}")
