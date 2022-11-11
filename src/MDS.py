import numpy as np
from sklearn.manifold import MDS
import data_loader
import matplotlib.pyplot as plt 

class Pre_Cluster_MultiDimensionalScaling:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.labels = None

    def generate_data_centers(self, data, labels):
        pre_MDS_centers = np.zeros((self.num_classes, data.shape[1]))
        for i in range(self.num_classes):
            data_assoc_with_labels = data[np.argwhere(labels == i)[:,0]]
            center_i = np.mean(data_assoc_with_labels, axis = 0)
            pre_MDS_centers[i,:] = center_i
        return pre_MDS_centers


    def generate_labels(self, sample_data, sample_labels):
        data = sample_data.reshape((sample_data.shape[0],np.prod(sample_data.shape[1:])))
        pre_MDS_centers = self.generate_data_centers(data, sample_labels)
        embedding = MDS(n_components = 100)
        centers_transformed = embedding.fit_transform(pre_MDS_centers)

        return 0

    def fetch_labels(self):
        return self.labels



if __name__ == "__main__":
    PCMS = Pre_Cluster_MultiDimensionalScaling(num_classes = 100)
    train_tuple, test_tuple = data_loader.load_cifar100()
    train_data, train_labels = train_tuple
    PCMS.generate_labels(train_data, train_labels)
    
