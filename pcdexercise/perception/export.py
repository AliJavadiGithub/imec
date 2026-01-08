import csv,json
import numpy as np

def export_csv(tracks,path):
    with open(path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(["id","t","x","y","z"])
        for t in tracks:
            for i,p in enumerate(t.traj):
                w.writerow([t.id,i,p[0],p[1],p[2]])

def export_json(tracks,path):
    out={t.id:np.array(t.traj).tolist() for t in tracks}
    with open(path,'w') as f: json.dump(out,f)
