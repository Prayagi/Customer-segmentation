import matplotlib.pyplot as plt
import seaborn as sns

def plot_elbow(inertia):
    plt.figure()
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def plot_clusters(data, labels):
    plt.figure()
    sns.scatterplot(x=data[:, 1], y=data[:, 2], hue=labels, palette='viridis')
    plt.title('Customer Segments')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.show()