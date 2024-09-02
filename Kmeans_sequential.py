from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import gc
#aux functions
def plot_initial_samples(X, y, title, hue_order, palette):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X[:, 0],y=X[:, 1],hue=y,
        palette=palette,hue_order=hue_order,legend='full',alpha=0.1
    )
    plt.title(title)
    plt.legend(title='Cover Type')
    plt.show()
def data_prep():
    # Data Preparation: Forest Cover Type Dataset
    forest_cover_data = fetch_ucirepo(id=31).data.original[
        ['Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Roadways', 'Cover_Type']]
    forest_cover_data = forest_cover_data.drop_duplicates()
    forest_cover_data.reset_index(drop=True, inplace=True)
    forest_cover_data.loc[forest_cover_data['Cover_Type'] < 2, 'Cover_Type_letters'] = 'C'
    forest_cover_data.loc[(forest_cover_data['Cover_Type'] == 2), 'Cover_Type_letters'] = 'B'
    forest_cover_data.loc[forest_cover_data['Cover_Type'] > 2, 'Cover_Type_letters'] = 'A'
    X = forest_cover_data[['Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Roadways']]
    y = forest_cover_data[['Cover_Type_letters']].values.flatten()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)
    dataset_sizes = [1, 0.5, 0.25]

    hue_order = sorted(y)
    palette = sns.color_palette("husl", len(hue_order))

    for size in dataset_sizes:
        if size < 1:
            rng = np.random.default_rng(seed=15)
            sample_indices = rng.choice(scaled_features.shape[0], size=int(size * scaled_features.shape[0]),
                                        replace=False)
            sample_data = scaled_features[sample_indices]
            sample_y = y[sample_indices]
        else:
            sample_data = scaled_features
            sample_y = y

        plot_initial_samples(sample_data, sample_y, f'Initial Data Plot ({size * 100}% of Dataset)', hue_order, palette)
        with open('dataset_size' + str(size * 100) + '.pkl', 'wb') as file:
            pickle.dump(sample_data, file)
def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
class K_Means(object):
    def __init__(self, k):
        self.n_clusters = k   # k
        self.clusters = {}     # empty clusters
        self.centroids = {}

    def initiate_centroids(self, X):
        idx = np.random.default_rng(seed=15).choice(len(X), self.n_clusters, replace=False)  # random position
        self.centroids = {kk: X[idx[kk]] for kk in range(self.n_clusters)}

    def euclidean_distance(self, P, Q):
        #return np.sqrt(((P - Q) ** 2).sum())
        return np.linalg.norm(P - Q)

    def nearest_centroid(self, point):
        distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids.values()]
        return np.argmin(distances)

    def recomputes_clusters(self, X):
        self.clusters = {i: [] for i in range(self.n_clusters)}
        for point in X:
            self.clusters[self.nearest_centroid(point)].append(point)

    def recomputes_centroids(self):
        for i in range(self.n_clusters):
            # Finds the average of the cluster at given index
            self.centroids[i] = np.mean(self.clusters[i], axis=0)


    def has_converged(self, old_centroids, tolerance=0.0001):
        for i in range(self.n_clusters):
            if self.euclidean_distance(self.centroids[i], old_centroids[i]) > tolerance:
                return False
        return True

    def plot_clusters(self):
        plt.figure(figsize=(8, 6))
        for i in range(self.n_clusters):
            cluster = np.array(self.clusters[i])
            sns.scatterplot(x=cluster[:, 0], y=cluster[:, 1], label=f'Cluster {i + 1}')
            plt.scatter(self.centroids[i][0], self.centroids[i][1], c='black', s=200, marker='X')
        plt.title('Cluster Plot')
        plt.legend()
        plt.show()

    def start(self,X):
        self.initiate_centroids(X)
        #print(self.centroids)
        end= False
        while not end:
            self.recomputes_clusters(X)
            old_centroids=self.centroids.copy()
            self.recomputes_centroids()
            end = self.has_converged(old_centroids)


if __name__ == '__main__':
    #only 1 run time if needed to generate pickle files
    #data_prep()
    file_names = ['dataset_size25.0.pkl','dataset_size50.0.pkl','dataset_size100.pkl']

    for file_name in file_names:
        data = load_pickle_file(file_name)
        #sequential mode
        kmeans = K_Means(k=3)
        start_time = time.time()
        kmeans.start(data)
        time_diff = (time.time() - start_time)
        print(f"Time taken for {file_name }: {time_diff:.6f} seconds.")
        #kmeans.plot_clusters()

        # Clean up after sequential part
        del kmeans
        gc.collect()  # Force garbage collection