"""Export all SUMO arrays to readable CSV and verify every value against NPZ."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIR_KEYS = ('front_xy_m', 'center_xy_m', 'rear_xy_m')
SCALAR_KEYS = ('heading_rad', 'speed_mps', 'acceleration_mps2', 'lane_id', 'lane_position_m')
STATIC_KEYS = ('vehicle_length_m', 'vehicle_width_m', 'vehicle_height_m')
FIELDS = ['frame_index','vehicle_index','vehicle_id','time_s','simulation_time_s'] + [k.replace('_xy_m',axis+'_m') for k in PAIR_KEYS for axis in ('_x','_y')] + list(SCALAR_KEYS)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def export(n):
    folder = ROOT / f'resource_allocation/n{n}/data'
    with np.load(folder/'mobility.npz', allow_pickle=False) as z: data={k:z[k] for k in z.files}
    with np.load(folder/'buildings.npz', allow_pickle=False) as z: boxes=z['boxes_xyz_m']
    out=folder/'csv'; out.mkdir(exist_ok=True)
    frames=len(data['time_s'])
    rows=[]
    for t in range(frames):
        for i in range(n):
            row=[t,i,str(data['vehicle_ids'][i]),float(data['time_s'][t]),float(data['simulation_time_s'][t])]
            row.extend(float(data[k][t,i,a]) for k in PAIR_KEYS for a in (0,1))
            row.extend(str(data[k][t,i]) if k=='lane_id' else float(data[k][t,i]) for k in SCALAR_KEYS)
            rows.append(row)
    def write(name,header,values):
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f); w.writerow(header); w.writerows(values)
    chunks=[]
    for start in range(0,len(rows),2000):
        name=f'mobility_rows_{start:06d}_{min(start+2000,len(rows))-1:06d}.csv'
        write(name,FIELDS,rows[start:start+2000]); chunks.append(name)
    write('mobility_preview.csv',FIELDS,rows[:min(n*2,200)])
    write('vehicles.csv',['vehicle_index','vehicle_id',*STATIC_KEYS],[[i,str(data['vehicle_ids'][i]),*[float(data[k][i]) for k in STATIC_KEYS]] for i in range(n)])
    write('buildings.csv',['building_index','x_min_m','y_min_m','z_min_m','x_max_m','y_max_m','z_max_m'],[[i,*map(float,b)] for i,b in enumerate(boxes)])
    # Rebuild arrays from persisted CSV, including IDs and every floating-point field.
    restored={k:np.empty_like(v) for k,v in data.items()}
    seen=set()
    for name in chunks:
        with (out/name).open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                t,i=int(row['frame_index']),int(row['vehicle_index'])
                if (t,i) in seen: raise ValueError('Duplicate row')
                seen.add((t,i)); restored['vehicle_ids'][i]=row['vehicle_id']
                for key in ('time_s','simulation_time_s'): restored[key][t]=float(row[key])
                for key in PAIR_KEYS:
                    for a,axis in enumerate(('_x','_y')): restored[key][t,i,a]=float(row[key.replace('_xy_m',axis+'_m')])
                for key in SCALAR_KEYS: restored[key][t,i]=row[key] if key=='lane_id' else float(row[key])
    with (out/'vehicles.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            i=int(row['vehicle_index'])
            if row['vehicle_id']!=data['vehicle_ids'][i]: raise ValueError('ID mismatch')
            for key in STATIC_KEYS: restored[key][i]=float(row[key])
    with (out/'buildings.csv').open(encoding='utf-8',newline='') as f:
        restored_boxes=np.array([[float(row[k]) for k in ['x_min_m','y_min_m','z_min_m','x_max_m','y_max_m','z_max_m']] for row in csv.DictReader(f)])
    mismatches=[k for k in data if not np.array_equal(data[k],restored[k])]
    if not np.array_equal(boxes,restored_boxes): mismatches.append('boxes_xyz_m')
    if mismatches or len(seen)!=frames*n: raise ValueError('Roundtrip mismatch: '+str(mismatches))
    report={'n':n,'frames':frames,'vehicle_time_rows':len(rows),'csv_chunks':chunks,'original_sha256':{name:sha(folder/name) for name in ['mobility.npz','buildings.npz']},'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in data.items()},'csv_sha256':{p.name:sha(p) for p in sorted(out.glob('*.csv'))},'roundtrip_exact':True,'mismatches':mismatches}
    (folder/'csv_export_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'n':n,'rows':len(rows),'roundtrip_exact':True}

if __name__=='__main__': print(json.dumps([export(n) for n in (10,50,100)]))
