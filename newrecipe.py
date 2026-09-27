from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

print("Reading Data")
data = pd.read_csv("Inputs/full_dataset.csv").sample(800000)

data.drop(["link", "source", "NER", "Unnamed: 0"], axis=1, inplace=True)

print("Vectorizing.....")
# Feature Extraction
# Convert the preprocessed text data into numerical feature vectors
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(data["title"])
y = data["ingredients"]
del data
print("Feature reducing ...")
svd = TruncatedSVD(n_components=100)
X_svd = svd.fit_transform(X)
joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(svd, "svd_model.joblib")
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_svd, y, test_size=0.2, random_state=42
)

# Split data into batches
batch_size = 2500
num_batches = int(np.ceil(X_train.shape[0] / batch_size))

# Initialize model BernoulliNB
model = BernoulliNB()  # joblib.load("model.pkl")

# Train the model in batches
for i in tqdm(range(num_batches)):
    print(f"\nBatch: {i}\n")
    X_batch = X_train[i * batch_size : (i + 1) * batch_size]
    y_batch = y_train[i * batch_size : (i + 1) * batch_size]
    model.partial_fit(X_batch, y_batch, classes=np.unique(y))

y_pred_list = []
test_batch_size = 2500

for i in tqdm(range(int(np.ceil(X_test.shape[0] / test_batch_size)))):
    print(f"\nEvaluating Batch: {i}\n")
    X_test_batch = X_test[i * test_batch_size : (i + 1) * test_batch_size]
    y_pred_batch = model.predict(X_test_batch)
    y_pred_list.append(y_pred_batch)

y_pred = np.concatenate(y_pred_list, axis=0)
y_pred = pd.DataFrame({"ingredients": y_pred})
print(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="micro")
recall = recall_score(y_test, y_pred, average="micro")
f1 = f1_score(y_test, y_pred, average="micro")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

joblib.dump(model, "model.pkl")

# Make new predictions
dish_name = "Katsu Chicken Bowl"
dish_name_vec = vectorizer.transform([dish_name])
dish_pca = svd.transform(dish_name_vec)
predicted_ingredients = model.predict(dish_pca)

print(predicted_ingredients)
X_svd_df = pd.DataFrame(X_svd)
y_df = pd.DataFrame(y)

# Optionally, you can add column names to the dataframes
X_svd_df.columns = ["X_svd_" + str(i) for i in range(X_svd.shape[1])]
y_df.columns = ["y"]

# Save the dataframes as CSV files
X_svd_df.to_csv("X_svd.csv", index=False)
y_df.to_csv("y.csv", index=False)


### EXAMPLE OF PREDICTION CODE ###

import joblib
import pandas as pd

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.joblib")
svd = joblib.load("svd_model.joblib")


def predict(dish_name):
    dish_name_vec = vectorizer.transform([dish_name])
    dish_pca = svd.transform(dish_name_vec)
    predicted_ingredients = model.predict(dish_pca)
    return predicted_ingredients


for dish_name in [
    "Kangnam Wings",
    "Fire Bowl",
    "Portabella Burger",
    "BBQ Chicken Bowl",
]:
    print(predict(dish_name))
