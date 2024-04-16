import numpy as np

from ge.classify import read_node_label, Classifier
from ge import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings):
    X, Y = read_node_label(
        r"C:\Users\al105\OneDrive\Desktop\OU\Thesis\Line_code\data\prof.txt"
    )
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(
    embeddings,
):
    X, Y = read_node_label(
        r"C:\Users\al105\OneDrive\Desktop\OU\Thesis\Line_code\data\prof.txt"
    )
    # print(f'{X}, {Y}')
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)
    print(emb_list)
    model = TSNE(
        n_components=2, perplexity=2
    )  # changed perplexity = 2 here since we have small dataset
    node_pos = model.fit_transform(emb_list)
    print(node_pos)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        print(f"C={c}, idx={idx}")
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


def plot_embeddings2(
    embeddings,
):
    X, Y = read_node_label(
        r"C:\Users\al105\OneDrive\Desktop\OU\Thesis\Line_code\data\prof.txt"
    )
    # print(f'{X}, {Y}')
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)
    print(emb_list)
    model = TSNE(
        n_components=2, perplexity=2
    )  # changed perplexity = 2 here since we have small dataset
    node_pos = model.fit_transform(emb_list)
    # print(node_pos)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for i in range(len(node_pos)):
        plt.scatter(emb_list[i][0], emb_list[i][1], label=i)

    #    for c, idx in color_idx.items():
    #        print(f'C={c}, idx={idx}')
    #        plt.scatter(emb_list[idx, 0], emb_list[idx, 1], label=idx)
    plt.legend()
    plt.show()


def plot_embeddings3(embeddings):
    X, Y = read_node_label(
        r"C:\Users\al105\OneDrive\Desktop\OU\Thesis\Line_code\data\prof.txt"
    )
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)
    # model = TSNE(n_components=2, perplexity=2)  # changed perplexity = 3 here
    # node_pos = model.fit_transform(emb_list)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    # Scatter plot
    for i in range(len(node_pos)):
        ax.scatter(node_pos[i, 0], node_pos[i, 1], node_pos[i, 2], label=i, s=50)

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


if __name__ == "__main__":
    G = nx.read_edgelist(
        r"C:\Users\al105\OneDrive\Desktop\OU\Thesis\Line_code\data\prof_edges.txt",
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[("weight", int)],
    )

    orders = ["first", "second", "all"]
    for order in orders:
        model = LINE(
            G, embedding_size=2, order=order
        )  # TODO remember to change per plot
        model.train(batch_size=10, epochs=50, verbose=2)
        embeddings = model.get_embeddings()

        with open(
            f"C:\\Users\\al105\\OneDrive\\Desktop\\OU\\Thesis\\Line_code\\prof_embeddings\\2d_TSNE_embeddings_{order}.txt",
            "w",
        ) as file:
            for node, embedding in embeddings.items():
                embedding_str = " ".join(map(str, embedding))
                file.write(f"{node} {embedding_str}\n")

        evaluate_embeddings(embeddings)
        plot_embeddings3(embeddings)
