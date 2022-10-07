import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from keras.datasets import fashion_mnist
from sklearn.metrics.pairwise import distance_metrics
from sklearn.decomposition import KernelPCA
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler

# Fashion-MNIST 데이터 불러오기
def load_data(n_samples):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    X = images = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    # 데이터 2차원 array로 변환
    (num_samples, num_feat1, num_feat2) = X.shape
    X = X.reshape(num_samples, -1)
    X = X[:n_samples, :]
    y = y[:n_samples]
    images = images[:n_samples, :]

    label_dict = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    return X, y, images, label_dict


# k-graph 설정
def set_graph(X: np.array, k: int = None):
    assert k != None, "Type the number of neighbors"
    euclidean = distance_metrics()["euclidean"]
    dist_matrix = euclidean(X)
    nodes = [i for i in range(len(X))]
    edges_with_weights = []
    for i in range(len(X)):
        dist_list = dist_matrix[i]
        dist_list_sorted = sorted(dist_list)
        neighbors = []
        for dist in dist_list_sorted[:k]:
            v = np.where(dist_list == dist)[0][0]
            if v != i:
                neighbors.append(v)
        for j in neighbors:
            edges_with_weights.append((i, j, dist_matrix[i][j]))
    G = nx.Graph()
    G.add_weighted_edges_from(edges_with_weights)

    return G, dist_matrix, nodes


# 다익스트라 알고리즘
def dijkstra(G, initial_node):
    visited_dist = {initial_node: 0}
    nodes = set(G.nodes())
    while nodes:
        connected_node = None
        for node in nodes:
            if node in visited_dist:
                if connected_node is None:
                    connected_node = node
                elif visited_dist[node] < visited_dist[connected_node]:
                    connected_node = node
        if connected_node is None:
            break
        nodes.remove(connected_node)
        cur_wt = visited_dist[connected_node]
        for v in dict(G.adj[connected_node]).keys():
            wt = cur_wt + dict(G.adj[connected_node])[v]["weight"]
            if v not in visited_dist or wt < visited_dist[v]:
                visited_dist[v] = wt
    return visited_dist


# 다익스트라 알고리즘을 이용하여 노드 간 최단 거리 구하기
def set_distances(G, nodes):
    N = len(nodes)
    final_dist_matrix = np.zeros((N, N))
    for node in nodes:
        visited_dist = dijkstra(G, node)
        for v, dist in visited_dist.items():
            final_dist_matrix[node][v] += dist
    return final_dist_matrix


# KernelPCA를 이용한 ISOMAP 구현
def isomap(distance_matrix, n_components, PCA_config):
    kernel_pca_ = KernelPCA(n_components, **PCA_config)
    Z = (distance_matrix**2) * (-0.5)
    embedding = kernel_pca_.fit_transform(Z)
    return embedding


def plot_embedding(args, X, y, label_dict, images):

    _, ax = plt.subplots(figsize=(15, 20))

    X = MinMaxScaler().fit_transform(X)

    for target_int in label_dict.keys():
        ax.scatter(
            *X[y == target_int].T,
            marker=f"${label_dict[target_int]}$",
            s=1000,
            color=plt.cm.Dark2(target_int),
            alpha=0,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every clothes on the embedding
        # show an annotation box for a group of cloth-types
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(args.title, color="Black", fontsize=20)
    save_path = os.path.join(args.fig_folder, f"{args.title}.png")
    ax.figure.savefig(save_path)
    print("Finished saving the output figure")
    ax.axis("off")


class Parser:
    def __init__(
        self,
        description: str,
    ) -> None:
        self.description = description

    def set_parser(self):
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument(
            "--num_samples",
            default=100,
            type=int,
            help="Number of samples to check",
        )
        parser.add_argument(
            "--n_neighbors",
            default=10,
            type=int,
            help="Number of neighbor for making k-graph",
        )
        parser.add_argument(
            "--fig_folder",
            default="figure",
            help="Save path for saving output figures",
        )
        parser.add_argument(
            "--title",
            default="ISOMAP Projection figure",
            type=str,
            help="Save path for saving output figures",
        )
        return parser.parse_args()


def main(args):
    print("Start!")
    X, y, images, label_dict = load_data(args.num_samples)
    print("Finished getting the data")
    G, origin_dist_matrix, nodes = set_graph(X, args.n_neighbors)
    final_dist_matrix = set_distances(G, nodes)
    print("Finished getting the distance matrix")
    pca_config = dict(
        kernel="precomputed", eigen_solver="auto", random_state=42, n_jobs=-1
    )
    isomap_config = dict(
        distance_matrix=final_dist_matrix, n_components=2, PCA_config=pca_config
    )

    isomap_embedding = isomap(**isomap_config)
    plot_embedding(args, isomap_embedding, y, label_dict, images)


if __name__ == "__main__":
    args = Parser("ISOMAP").set_parser()
    if not os.path.exists(args.fig_folder):
        os.makedirs(args.fig_folder)
    main(args)
