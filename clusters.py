from sklearn.cluster import KMeans
from model_weights import extract_model_weights
from sklearn.preprocessing import normalize
from cosine_similarity import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


# Function to perform clustering with different techniques
def create_clusters(models, n_clusters, model_name):
    # Extract weights from each model and convert them to numpy arrays for sklearn functions
    model_weights = [extract_model_weights(model).cpu().numpy() for model in models]

    # Normalize the weights to unit vectors for cosine similarity
    model_weights_normalized = normalize(model_weights)

    # Perform clustering based on the specified model name
    if model_name == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++')
        kmeans.fit(model_weights_normalized)
        labels = kmeans.labels_
    elif model_name == 'dbscan':
        dbscan = DBSCAN()
        labels = dbscan.fit_predict(model_weights_normalized)
    elif model_name == 'agglomerative':
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(model_weights_normalized)
    elif model_name == 'mean_shift':
        mean_shift = MeanShift()
        labels = mean_shift.fit_predict(model_weights_normalized)
    elif model_name == 'gmm':
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(model_weights_normalized)
        labels = gmm.predict(model_weights_normalized)
    elif model_name == 'spectral':
        spectral = SpectralClustering(n_clusters=n_clusters)
        labels = spectral.fit_predict(model_weights_normalized)
    else:
        raise ValueError("Unsupported model name. Choose from 'kmeans', 'dbscan', 'agglomerative', 'mean_shift', 'gmm', 'spectral'.")

    # Group models by their cluster labels
    clusters = {i: [] for i in range(n_clusters)}
    for model, label in zip(models, labels):
        clusters[label].append(model)

    return clusters

def create_positive_cluster(clusters, local_model):
    # Extract and normalize weights of the local model
    local_weights = extract_model_weights(local_model)
    local_weights_normalized = normalize([local_weights])[0]  

    min_sim = float('inf')
    positive_cluster_key = None

    # Iterate through each cluster and each model within those clusters
    for cluster_key, cluster in clusters.items():
        for model in cluster:
            # Extract and normalize weights of the current model
            model_weights = extract_model_weights(model)
            model_weights_normalized = normalize([model_weights])[0]  # Uncomment if normalization is needed

            # Calculate cosine similarity
            sim = cosine_similarity(local_weights, model_weights)

            # Update the minimum similarity and corresponding cluster if necessary
            if sim < min_sim:
                min_sim = sim
                positive_cluster_key = cluster_key

    # Ensure that a positive cluster key was found
    if positive_cluster_key is not None:
        positive_cluster = clusters.pop(positive_cluster_key, {})
    else:
        positive_cluster = {}

    return positive_cluster, clusters