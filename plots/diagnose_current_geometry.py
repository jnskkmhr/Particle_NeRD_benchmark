import json
from pathlib import Path

import h5py
import numpy as np

run = Path('/home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/g1_humanoid/data/current-20261109')
m = json.loads((run / 'metadata.json').read_text())
d = np.load(run / 'trace.npz')

def rotate(q, v):
    t = 2 * np.cross(q[:3], v)
    return v + q[3] * t + np.cross(q[:3], t)

def multiply(q, r):
    return np.r_[q[3]*r[:3]+r[3]*q[:3]+np.cross(q[:3],r[:3]), q[3]*r[3]-np.dot(q[:3],r[:3])]

rows = []
with h5py.File(run/'states.h5') as s, h5py.File(m['source_recording']) as f:
    print('Foot IDs:',f['foot_body_ids'][:], 'Collider IDs:',f['collider_body'][:])
    print('HDF keys:', list(f.keys()))
    feet = f['foot_body_ids'][:]
    for t in [0, .2, .5, .8, 1, 1.5, 2, 2.2, 2.5, 3, 4, 5]:
        k = round(t * 50); source_k = m['start_frame'] + k*4
        row={'t':t,'feet':[]}
        for i,body in enumerate(feet):
            p=s['body_pose'][k,body]; local=f['collider_local_pose'][i]; half=f['collider_half_size'][i]
            center=p[:3]+rotate(p[3:],local[:3]);q=multiply(p[3:],local[3:]);inv=np.r_[-q[:3],q[3]]
            v=f['body_velocity'][source_k,body]
            speed=np.sqrt(np.sum(v[:3]**2)+np.sum(v[3:]**2*np.array([half[1]**2+half[2]**2,half[0]**2+half[2]**2,half[0]**2+half[1]**2])))
            foot={'center_z':float(center[2]),'speed':float(speed),'vz':float(v[2])}
            if k:
                foot['nerd_Fz']=float(d['forces'][k*4-1,i,2]);foot['mpm_Fz']=float(d['reference_forces'][k*4-1,i,2])
            for key in ['particle_position','reference_position']:
                x=s[key][k];z=rotate(inv,x-center)
                footprint=(np.abs(z[:,0])<=half[0]) & (np.abs(z[:,1])<=half[1])
                below=footprint & (z[:,2]<-half[2]);above=footprint & (z[:,2]>half[2])
                foot[key]={'under_count':int(below.sum()),'under_within_margin':int((below & (z[:,2]>=-half[2]-.02)).sum()),'over_within_margin':int((above & (z[:,2]<=half[2]+.02)).sum()),'inside_box':int((footprint & (np.abs(z[:,2])<=half[2])).sum()),'nearest_below_gap_mm':float((-half[2]-z[below,2]).min()*1000) if below.any() else None}
            row['feet'].append(foot)
        rows.append(row)
for row in rows:print(json.dumps(row))
for lo,hi in [(0,.5),(.8,2),(2.5,5)]:
    mask=(d['time']>=lo)&(d['time']<=hi)
    print('MEANS',lo,hi,'NeRD',d['forces'][mask,:,2].mean(0),'MPM',d['reference_forces'][mask,:,2].mean(0))
Path('/tmp/nerd-force-geometry.json').write_text(json.dumps(rows,indent=2))
