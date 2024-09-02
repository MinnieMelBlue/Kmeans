#from ucimlrepo import fetch_ucirepo
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pickle
import time
import gc

from multiprocessing import Pool,cpu_count


#aux functions
# def plot_initial_samples(X, y, title, hue_order, palette):
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(
#         x=X[:, 0],y=X[:, 1],hue=y,
#         palette=palette,hue_order=hue_order,legend='full',alpha=0.1
#     )
#     plt.title(title)
#     plt.legend(title='Cover Type')
#     plt.show()
# def data_prep():
#     # Data Preparation: Forest Cover Type Dataset
#     forest_cover_data = fetch_ucirepo(id=31).data.original[
#         ['Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Roadways', 'Cover_Type']]
#     forest_cover_data = forest_cover_data.drop_duplicates()
#     forest_cover_data.reset_index(drop=True, inplace=True)
#     forest_cover_data.loc[forest_cover_data['Cover_Type'] < 2, 'Cover_Type_letters'] = 'C'
#     forest_cover_data.loc[(forest_cover_data['Cover_Type'] == 2), 'Cover_Type_letters'] = 'B'
#     forest_cover_data.loc[forest_cover_data['Cover_Type'] > 2, 'Cover_Type_letters'] = 'A'
#     X = forest_cover_data[['Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Roadways']]
#     y = forest_cover_data[['Cover_Type_letters']].values.flatten()
#
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(X)
#     dataset_sizes = [1, 0.5, 0.25]
#
#     hue_order = sorted(y)
#     palette = sns.color_palette("husl", len(hue_order))
#
#     for size in dataset_sizes:
#         if size < 1:
#             rng = np.random.default_rng(seed=15)
#             sample_indices = rng.choice(scaled_features.shape[0], size=int(size * scaled_features.shape[0]),
#                                         replace=False)
#             sample_data = scaled_features[sample_indices]
#             sample_y = y[sample_indices]
#         else:
#             sample_data = scaled_features
#             sample_y = y
#
#         plot_initial_samples(sample_data, sample_y, f'Initial Data Plot ({size * 100}% of Dataset)', hue_order, palette)
#         with open('dataset_size' + str(size * 100) + '.pkl', 'wb') as file:
#             pickle.dump(sample_data, file)

def plot_execution_time_vs_records(test_times_by_cores):
    plt.figure(figsize=(10, 6))

    for cores, times in test_times_by_cores.items():
        times_array = np.array(times)
        num_records = times_array[:, 0]  # Number of records
        execution_time = times_array[:, 1]  # Execution time

        plt.plot(num_records, execution_time, marker='o', linestyle='-', label=f'{cores} cores')

    plt.xlabel('Number of Records')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs. Number of Records for Different Number of Cores')
    plt.grid(True)
    # Optional: Logarithmic scales if needed
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.show()


def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

class K_Means(object):
    def __init__(self, k, n_cores=None):
        self.n_clusters = k
        self.clusters = {}
        self.centroids = {}
        self.data = None
        self.n_cores = n_cores if n_cores else cpu_count()

    def set_data(self, X):
        self.data = X
    def initiate_centroids(self):
        idx = np.random.default_rng(seed=15).choice(len(self.data), self.n_clusters, replace=False)
        self.centroids = {kk: self.data[idx[kk]] for kk in range(self.n_clusters)}

    def euclidean_distance(self, P, Q):
        return np.sqrt(((P - Q) ** 2).sum())
        #return np.linalg.norm(P - Q)

    def nearest_centroid(self, point):
        distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids.values()]
        return np.argmin(distances)

    def recomputes_centroids(self):
        old_centroids = self.centroids.copy()
        for i in range(self.n_clusters):
            if len(self.clusters[i]) > 0:
                self.centroids[i] = np.mean(self.clusters[i], axis=0)
        return old_centroids

    def has_converged(self, old_centroids, tolerance=0.0001):
        for i in range(self.n_clusters):
            if self.euclidean_distance(self.centroids[i], old_centroids[i]) > tolerance:
                return False
        return True

    # def plot_clusters(self):
    #     plt.figure(figsize=(8, 6))
    #     for i in range(self.n_clusters):
    #         cluster = np.array(self.clusters[i])
    #         sns.scatterplot(x=cluster[:, 0], y=cluster[:, 1], label=f'Cluster {i + 1}')
    #         plt.scatter(self.centroids[i][0], self.centroids[i][1], c='black', s=200, marker='X')
    #     plt.title('Cluster Plot')
    #     plt.legend()
    #     plt.show()

    def assign_chunk(self,chunk):
        return [self.nearest_centroid(point) for point in chunk]

    def iterative_part(self):
        end = False
        while not end:
            #reset clusters
            self.clusters = {i: [] for i in range(self.n_clusters)}

            chunks = np.array_split(self.data, self.n_cores)

            with Pool(self.n_cores) as pool:
                results = pool.map(self.assign_chunk, chunks)

            labels = [label for result in results for label in result]

            for point, label in zip(self.data, labels):
                self.clusters[label].append(point)


            old_centroids = self.recomputes_centroids()
            end = self.has_converged(old_centroids)





def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

if __name__ == '__main__':
    #only 1 run time if needed to generate pickle files
    #data_prep()
    file_names = ['dataset_size25.0.pkl','dataset_size50.0.pkl','dataset_size100.pkl']
    ncores=[2,4,6,8,12]

    #file_name=file_names[2]
    #cores=ncores[2]
    test_times_by_cores = {cores: [] for cores in ncores}

    for file_name in file_names:
        data = load_pickle_file(file_name)
        len_data=len(data)
        print(len_data)

        #parallel mode
        for cores in ncores:
            kmeanss = K_Means(k=3, n_cores=cores)
            kmeanss.set_data(data)
            kmeanss.initiate_centroids()
            start_time = time.time()
            kmeanss.iterative_part()
            time_diff = (time.time() - start_time)
            print(f"Time taken for {file_name } with  {cores} cores: {time_diff:.6f} seconds.")
            #kmeanss.plot_clusters()
            test_times_by_cores[cores].append([len_data, time_diff])

    plot_execution_time_vs_records(test_times_by_cores)
    # Clean up
    # del kmeanss
    # gc.collect()  # Force garbage collection
    #
    # # Clean up
    # del data
    # gc.collect()  # Force garbage collection


