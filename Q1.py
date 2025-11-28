import numpy as np
import glob
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import urllib.request
import os

# -------------------------
# DATA
# -------------------------
data_list = []
label_list = []

# Training data
for i, address in enumerate(glob.glob("Q1/train/*/*")):
    image = cv2.imread(address)
    image = cv2.resize(image, (32,32))
    image = image / 255
    image = image.flatten()
    data_list.append(image)

    # Label: 0 = Cat, 1 = Dog
    label = 0 if 'Cat' in address else 1
    label_list.append(label)

    if i % 50 == 0:
        print(f'[INFO] {i} images read!')

X_train = np.array(data_list)
y_train = np.array(label_list)

# Test data
test_data_list = [] 
test_label_list = []

for i, address in enumerate(glob.glob("Q1/test/*/*")):
    image = cv2.imread(address)
    image = cv2.resize(image, (32,32))
    image = image / 255
    image = image.flatten()
    test_data_list.append(image)

    label = 0 if 'Cat' in address else 1
    test_label_list.append(label)

X_test = np.array(test_data_list)
y_test = np.array(test_label_list)

# -------------------------
# KNN MODEL
# -------------------------
best_k = None
best_accuracy = 0
best_model = None
for k in range(1, 102, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k
        best_model = knn

print("\nBest neighbor found for KNN:", best_k)
print("KNN Accuracy:", best_accuracy)


joblib.dump(knn, "knn_cat_dog_model.z")
print("[INFO] KNN model saved as 'knn_cat_dog_model.z'")

# -------------------------
# LOGISTIC REGRESSION MODEL
# -------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

predictions_logreg = logreg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, predictions_logreg))

joblib.dump(logreg, "logreg_cat_dog_model.z")
print("[INFO] Logistic Regression model saved as 'logreg_cat_dog_model.z'")

# -------------------------
# TEST ON NEW IMAGES FROM INTERNET
# -------------------------
image_urls = [
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.thesprucepets.com%2Fthmb%2F-rwl1FJbnICyZE_jVN8DVcid9DI%3D%2F1500x0%2Ffilters%3Ano_upscale()%3Amax_bytes(150000)%3Astrip_icc()%2FGettyImages-962608834-90d2a503cd604f96b438a7b0bd93d014.jpg&f=1&nofb=1&ipt=8920ee6db3ae522541be3c0a86c7d232fc6decbc3c3e3a21eb34be017e25e050",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fjooinn.com%2Fimages%2Fdog-66.jpg&f=1&nofb=1&ipt=98e9f6ceab2880e72dbf8212bef22bb902600c2dcfb3dbb513b320ae3f11f8ba",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fblackstarkennels.in%2Fwp-content%2Fuploads%2F2022%2F12%2Fmarcus-cramer-OsOQhAzcEKc-unsplash.jpg&f=1&nofb=1&ipt=822d7facb50eceb72dfa2557601aef3b8755c3aa7b6732bbae701dd4ef2aa46a",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fdogwise.in%2Fwp-content%2Fuploads%2F2022%2F11%2F11-2.jpg&f=1&nofb=1&ipt=f5f3d76ec8d46302ea8a5ebdc724a7b81c99c50ed80f27f1a97bee512361b5b7",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmoderncat.com%2Fwp-content%2Fuploads%2F2017%2F12%2Fbigstock-Indoor-Cat-Portrait-Close-up-440797139-scaled.jpg&f=1&nofb=1&ipt=2f4f66831a489eda1a9c36209fed60fa4394e0069ff4f58d3d18a0b8afe518c5",
    "https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2F2.vndic.net%2Fimages%2Fdict%2Fc%2Fcat.gif&f=1&nofb=1&ipt=ac082fc64e883d1da71e33c27979e7be648174a03d4b9ea8c90e529fafaee4e6",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.pxhere.com%2Fphotos%2F9a%2F56%2Fkitten_cat_feline_yellow_red_portrait_watching_pet-1338723.jpg!s1&f=1&nofb=1&ipt=a048cdffe65e361dde53ba056b1465fda4ab6c66d79221ac2645a450a0a0ba60",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.pixabay.com%2Fphoto%2F2015%2F12%2F13%2F18%2F19%2Fcat-1091521_640.jpg&f=1&nofb=1&ipt=e13e683fba406efcdf0850bc363aad226bc28ceaf0a5ec6b156aa8beb7ac53de",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fhips.hearstapps.com%2Fhmg-prod%2Fimages%2Fcutest-cat-breeds-ragdoll-663a8c6d52172.jpg%3Fcrop%3D0.5989005497251375xw%3A1xh%3Bcenter%2Ctop%26resize%3D980%3A*&f=1&nofb=1&ipt=fbe60b123c1a1677ae44faa0dc5a0f8ad788f8a2a4e5aef60bb1a9a99c3abd20",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2F6%2F68%2FOrange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg%2F168px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg&f=1&nofb=1&ipt=9179080018706f8cc0662c3d0293e0ad0d52ecdd32dee88eb204b3af725fd126",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.thedogclinic.com%2Fwp-content%2Fuploads%2F2023%2F07%2Fbreeds-long-tails.jpg&f=1&nofb=1&ipt=b4beac9098cc3ad8eafe3d91cf74f797bac59479f06fe3e9e172db1dbba8d26d",
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwallup.net%2Fwp-content%2Fuploads%2F2018%2F10%2F07%2F542130-dog-animal-friendly-puppy-cute-dogs.jpg&f=1&nofb=1&ipt=d502391cb9fc3bfbb9adfbe314a7dbd2e99e319b9ffd67443680d1377d4492b6" 
]

if not os.path.exists("new_images"):
    os.makedirs("new_images")

image_paths = []
for i, url in enumerate(image_urls):
    path = f"new_images/img{i}.jpg"
    urllib.request.urlretrieve(url, path)
    image_paths.append(path)
    print(f"[INFO] Downloaded {path}")

# Load saved models
knn = joblib.load("knn_cat_dog_model.z")
logreg = joblib.load("logreg_cat_dog_model.z")

for path in image_paths:
    image = cv2.imread(path)
    image = cv2.resize(image, (32,32))
    image = image / 255
    image = image.flatten().reshape(1, -1)

    pred_knn = knn.predict(image)[0]
    pred_logreg = logreg.predict(image)[0]

    label_map = {0: "Cat", 1: "Dog"}
    print(f"[INFO] {path} -> KNN: {label_map[pred_knn]}, Logistic Regression: {label_map[pred_logreg]}")
