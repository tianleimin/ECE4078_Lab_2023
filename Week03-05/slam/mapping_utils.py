# for saving the SLAM map
import numpy as np
import json

class MappingUtils:
    def __init__(self, markers = np.zeros((2,0)), covariance = np.zeros((0,0)), taglist = []):
        self.markers = markers
        self.covariance = covariance
        self.taglist = taglist
        self.check_valid()
    
    def check_valid(self):
        assert (markers.shape[0] == 2), "Markers must be 2xn."
        n = markers.shape[1]
        assert (n == len(taglist)), "No. of markers and tags do not match."
        assert (2*n == self.covariance.shape[0]), "Covariance matrix does not match size of markers."
        assert (self.covariance.shape[0] == self.covariance.shape[1]), "Covariance matrix is not square."
        cov_sym_score = np.linalg.norm(self.covariance - self.covariance.T)
        assert(cov_sym_score < 0.01), "Covariance matrix is not symmetric."
    
    def save(self, fname="slam_map.txt"):
        map_attributes = {"taglist":self.taglist,
                          "markers":self.markers.tolist(),
                          "covariance":self.covariance.tolist()}
        with open(fname,'w') as map_file:
            json.dump(map_attributes, map_file, indent=2)
    
    def load(self, fname="slam_map.txt"):
        with open(fname,'r') as map_file:
            map_attributes = json.load(map_file)
        self.taglist = map_attributes["taglist"]
        self.markers = np.array(map_attributes["markers"])
        self.covariance = np.array(map_attributes["covariance"])
        self.check_valid()
    
    def compute_tf(self, other):
        markers1 = self.markers.copy()
        markers2 = other.markers.copy()

        # re-order the other markers to match self
        idxlist = [other.taglist.index(tag) for tag in self.taglist]
        markers2 = markers2[:, idxlist]

        armse, R, t = self.compute_armse(mark1, markers2)

        return armse, R, t

    @staticmethod
    def compute_armse(markers1, markers2):
        # An implementation of Umeyama's method
        n = markers1.shape[1]
        mu1 = 1.0/n * np.sum(markers1, axis=1, keepdims=True)
        mu2 = 1.0/n * np.sum(markers2, axis=1, keepdims=True)
        Sigma = 1.0/n * np.dot((markers2 - mu2), (markers1 - mu1).transpose())

        S = np.eye(2)

        U, D, VH = np.linalg.svd(Sigma)

        R = np.dot(U, np.dot(S, VH))
        t = mu2 - np.dot(R, mu1)

        errors = markers2 - np.dot(R,markers1) - t

        armse = np.sqrt( 1.0/n * np.sum(errors**2) )
        return armse, R, t

