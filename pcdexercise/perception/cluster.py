from sklearn.neighbors import NearestNeighbors
import numpy as np

def euclidean_clusters(pts, eps=0.35, minpts=10, maxpts=400):
    nn = NearestNeighbors(radius=eps).fit(pts)
    A = nn.radius_neighbors_graph(pts).toarray()
    visited = set()
    clusters = []
    for i in range(len(pts)):
        if i in visited: continue
        q = [i]; visited.add(i); comp=[]
        while q:
            j = q.pop()
            comp.append(j)
            for k in np.where(A[j])[0]:
                if k not in visited:
                    visited.add(k)
                    q.append(k)
        if minpts <= len(comp) <= maxpts:
            clusters.append(pts[comp])
    return clusters
