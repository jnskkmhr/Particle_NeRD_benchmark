"""Record one current integrated NeRD replay with synchronized reference snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import newton
import numpy as np
import torch
import warp as wp

from isaaclab_tasks.contrib.g1_29dof_particle_nerd.env_cfg.physics_cfg import g1_particle_nerd_physics_cfg
from isaaclab_tasks.contrib.particle_nerd.solver import ParticleNeRDSolver
from isaaclab_tasks.contrib.particle_nerd.solver_cfg import FixedBedCfg
from isaaclab_tasks.contrib.particle_nerd.network.inference import rotate, prepare_compact_step

p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
repo=Path('/home/jkamohara/isaac/IsaacLab-particle-nerd')
recording=Path('/home/jkamohara/isaac/particle_nerd/experiments/g1-evaluation-20s-mass50/reference-recordings/episode-0002.hdf5')
native_meta=Path('/home/jkamohara/isaac/particle_nerd/experiments/sinkage-diagnosis-20261028/dynamic-new-training/metadata.json')
feet_meta=json.loads(native_meta.read_text())['estimator']['feet']
torch.set_num_threads(2);wp.init();device='cuda:0';dt=.005;start=60
cfg=g1_particle_nerd_physics_cfg().solver_cfg.entries[1].solver_cfg
cfg.compile_model=False
code_root=repo/'source/isaaclab_tasks/isaaclab_tasks/contrib/particle_nerd'
hashes={str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in code_root.rglob('*.py')}
with h5py.File(recording) as f,torch.inference_mode():
 manifest=json.loads(f.attrs['manifest']);ids=f['collider_body'][:];feet=f['foot_body_ids'][:]
 assert len(ids)==7 and np.array_equal(ids[:2],feet)
 proxy=manifest['config']['sim']['physics']['solver_cfg']['proxies'][0]
 assert proxy['mode']=='lagged' and proxy['mass_scale']==1 and proxy['proxy_relaxation']==1
 names=['left_ankle_roll_link','right_ankle_roll_link','MPMBedFloor','MPMBedWallFront','MPMBedWallLeft','MPMBedWallRight','MPMApproachBank']
 builder=newton.ModelBuilder();builder.begin_world()
 for i,(body,name) in enumerate(zip(ids,names)):
  kwargs={'xform':wp.transform(f['body_pose'][start,body,:3],f['body_pose'][start,body,3:]),'label':'/World/'+name}
  if i<2:kwargs.update(mass=feet_meta[i]['virtual_mass_kg'],com=wp.vec3(feet_meta[i]['body_com_local_m']),inertia=wp.mat33(feet_meta[i]['virtual_inertia_local']),lock_inertia=True)
  b=builder.add_body(**kwargs);size=f['collider_half_size'][i]
  shape_cfg=builder.ShapeConfig(density=0,mu=float(f['collider_friction'][i]),margin=feet_meta[i]['shape_margin_m'] if i<2 else 0)
  builder.add_shape_box(b,xform=wp.transform(f['collider_local_pose'][i,:3],f['collider_local_pose'][i,3:]),hx=float(size[0]),hy=float(size[1]),hz=float(size[2]),cfg=shape_cfg)
 mass=f['particle_mass'][:];radius=f['particle_radius'][:];n=len(mass)
 builder.add_particles(f['particle_position'][start].tolist(),f['particle_velocity'][start].tolist(),mass.tolist(),radius.tolist())
 builder.end_world();model=builder.finalize(device=device)
 model.mpm=SimpleNamespace(**{k:wp.full(n,float(v),dtype=float,device=device) for k,v in manifest['material'].items() if k!='density'})
 solver=ParticleNeRDSolver(model,cfg);solver.bind_fixed_bed(FixedBedCfg())
 state=model.state();out=model.state()
 solver.history.copy_(torch.as_tensor(np.stack([f['particle_position'][j] for j in range(start-2,start+1)],axis=1),device=device))
 identity=wp.array(np.arange(7),dtype=int,device=device)
 mass_t=torch.tensor([x['virtual_mass_kg'] for x in feet_meta],device=device)
 inertia=torch.tensor([x['virtual_inertia_local'] for x in feet_meta],device=device)
 pose=f['body_pose'][start:start+a.steps+1];vel=f['body_velocity'][start:start+a.steps+1];ref_force=f['foot_material_wrench'][start:start+a.steps+1]
 output=h5py.File(a.output/'states.h5','w')
 snap_count=1+a.steps//4
 datasets={
  'particle_position':output.create_dataset('particle_position',(snap_count,n,3),dtype='f4'),
  'reference_position':output.create_dataset('reference_position',(snap_count,n,3),dtype='f4'),
  'body_pose':output.create_dataset('body_pose',(snap_count,pose.shape[1],7),dtype='f4'),
  'time_s':output.create_dataset('time_s',(snap_count,),dtype='f8'),
 }
 for key,val in [('particle_position',f['particle_position'][start]),('reference_position',f['particle_position'][start]),('body_pose',pose[0]),('time_s',0.)]:datasets[key][0]=val
 forces=[];metrics=[]
 print('CONFIG',n,solver.active_counts,cfg.checkpoint,flush=True)
 for k in range(a.steps):
  state.body_q.assign(pose[k,ids]);state.body_qd.assign(vel[k,ids]);solver.capture_collider_state(state,identity)
  q=torch.as_tensor(pose[k,feet,3:],device=device);iq=torch.cat((-q[:,:3],q[:,3:]),-1)
  old=torch.as_tensor(ref_force[k],device=device)
  local_torque=rotate(iq,old[:,3:]);angular=rotate(q,torch.linalg.solve(inertia,local_torque.unsqueeze(-1)).squeeze(-1))
  free=torch.as_tensor(vel[k+1,ids],device=device).clone()
  free[:2]-=dt*torch.cat((old[:,:3]/mass_t[:,None],angular),-1)
  wp.to_torch(state.body_qd).copy_(free)
  hist=torch.as_tensor(np.stack([f['particle_position'][j] for j in range(start+k-2,start+k+1)],axis=1),device=device)
  geom=solver.geometry[0]
  nh,pp,vv=solver.fixed_bed.inputs(hist,solver.neural_state,0,geom)
  prepared=prepare_compact_step(nh,pp,vv,geom,solver.materials[0],dt,support=solver.support,active_k=solver.active_counts[0],anchor_m=cfg.anchor_particles,reuse_selection=k%cfg.selection_interval!=0 and 'cached_topk' in geom)
  solver.estimator.capture(state,dt)
  out.particle_q.assign(f['particle_position'][start+k+1])
  out.particle_qd.assign((f['particle_position'][start+k+1]-f['particle_position'][start+k])/dt)
  solver.estimator.evaluate(out,active_idx=[geom['cached_topk']])
  x=wp.to_torch(out.particle_q);ref=torch.as_tensor(f['particle_position'][start+k+1],device=device)
  err=x-ref;near=((ref[:,:2,None]-torch.as_tensor(pose[k,feet,:2].T,device=device)[None])**2).sum(1).min(-1).values<.15**2
  metrics.append([float(err.square().sum(-1).mean().sqrt()),float(err[near,2].mean()),float(err[:,2].mean())])
  forces.append(solver.estimator.wrench.numpy().copy())
  assert torch.isfinite(x).all() and np.isfinite(forces[-1]).all()
  if (k+1)%4==0:
   j=(k+1)//4
   for key,val in [('particle_position',x.cpu().numpy()),('reference_position',ref.cpu().numpy()),('body_pose',pose[k+1]),('time_s',(k+1)*dt)]:datasets[key][j]=val
  state,out=out,state
  if k%100==0:print('STEP',k,'RMSE',metrics[-1][0],'Fz',float(forces[-1][:,2].sum()),flush=True)
 output.close()
 forces=np.asarray(forces);reference=ref_force[1:];time=np.arange(1,a.steps+1)*dt
 np.savez_compressed(a.output/'trace.npz',time=time,forces=forces,reference_forces=reference,metrics=np.asarray(metrics))
 metadata={'evaluation':'Force estimator only: prescribed MPM particles and recorded G1 motion; no neural prediction and no robot-force feedback','checkpoint':cfg.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(cfg.checkpoint).read_bytes()).hexdigest(),'code_sha256':hashes,'source_recording':str(recording),'start_frame':start,'dt_s':dt,'steps':a.steps,'video_hz':50,'particle_count':n,'active_k':solver.active_counts,'anchor_particles':cfg.anchor_particles,'selection_interval':cfg.selection_interval,'argon_foot_speed_gate':False,'network_forward_calls':0,'particle_input':'Recorded MPM positions; velocity = consecutive recorded position difference / dt; current Top-K selection on prescribed history','compile_model':False,'cuda_graph_capture':False,'material':manifest['material'],'bed_dimensions_m':manifest['bed_dimensions_m'],'reference_voxel_size_m':manifest['grid_spacing_m'],'force_parameters':{k:getattr(cfg,k) for k in ['effective_mass_scale','tangent_mass_scale','horizontal_mass_scale','upward_mass_scale','normal_power','contact_iterations','contact_relaxation','gradient_radius','gradient_contact_band','contact_stabilization','contact_offset']},'force_vector_rmse_N':float(np.sqrt(np.mean(np.sum((forces[:,:,:3]-reference[:,:,:3])**2,axis=-1)))),'final_particle_position_rmse_m':metrics[-1][0],'snapshots':'states.h5','snapshot_time_reference':'Elapsed replay time; source recording time = replay time + 0.300 s','reference_initialization':'Three recorded MPM particle-position frames; not an un-settled zero-warmup test'}
 for rel,h in hashes.items():assert hashlib.sha256((repo/rel).read_bytes()).hexdigest()==h,'Code changed during replay'
 (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print('DONE',metadata['force_vector_rmse_N'],flush=True)
