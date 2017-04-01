from bs4 import BeautifulSoup as soup
import pickle
import re
import numpy as np
import requests


def fetch(url):
    """download case from url"""
    return soup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, "lxml").text


def find_relevant_parts(raw, pattern):
    """apply pattern to retrieve legal discussion"""
    return re.findall(pattern, str(raw))


def encode(text, w2idx):
    """takes a list of (url, string) tuples and returns an encoding based on w2idx"""
    return np.array([w2idx.get(word.lower(), len(w2idx.keys()) + 1) for para in text[1] for word in para.split()])


def decode(encoding, idx2w):
    return ([idx2w[n] for n in encoding[1]])


def binarize(encoded, n):
    """takes the numerically encoded text and the number of unique words and returns a sparse matrix"""
    return np.array([1 if i in encoded else 0 for i in range(n)])


def load_transforms():
    """load the fitted PCA and clustering models"""
    with open("transforms.pkl", "rb") as f:
        (pca, kmeans) = pickle.load(f)
    return (pca, kmeans)


def main():
    """retrieve a case, encode and binarize, apply the fitted PCA, and predict cluster"""

    """build regex"""
    pattern = re.compile(".*\[\d\d\d\d\].*")

    """load vocab model"""
    with open("w2idx.pkl", "rb") as f:
        w2idx, _ = pickle.load(f)

    """fetch case"""
    url = "http://www.bailii.org/ew/cases/EWHC/QB/2015/620.html"
    text = fetch(url)

    """extract legal discussion"""
    print("retrieving relevant extracts...")
    extracts = (url, find_relevant_parts(text, pattern))

    """encode and binarize"""
    print("encoding...")
    encoded_case = encode(extracts, w2idx)
    binarized = binarize(encoded_case, len(w2idx.keys()))

    """load PCA and cluster models"""
    pca, kmeans = load_transforms()

    """apply feature reduction"""
    reduced = pca.transform(binarized[1].reshape(-1, 1))

    """predict cluster"""
    cluster = kmeans.predict(reduced)
    print("assigned to cluster {}".format(cluster))


if __name__ == "__main__":
    main()