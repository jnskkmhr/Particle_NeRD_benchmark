"""Render saved states only; no simulation advancement or regenerated trajectories."""
import sys
from pathlib import Path
frozen=Path('/home/jkamohara/isaac/particle_nerd/experiments/recurrent-05s-20261021/g1-source-stage/source/source')
for package in ('isaaclab_rl','isaaclab_assets','isaaclab_newton','isaaclab_tasks','isaaclab'):
 sys.path.insert(0,str(frozen/package))
sys.path.insert(0,'/home/jkamohara/isaac/IsaacLab-particle-nerd/examples')
import argparse,json,subprocess,hashlib
import h5py,numpy as np,warp as wp
from PIL import Image,ImageDraw,ImageFont
from isaaclab.app import launch_simulation
from isaaclab_newton.physics import NewtonManager
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from particle_nerd.g1.collect import configure

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args()
meta=json.loads((a.run/'metadata.json').read_text())
source=h5py.File(meta['source_recording']);snap=h5py.File(a.run/'states.h5')
task='IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play';cfg=configure(parse_env_cfg(task,num_envs=1),1)
# Restore the recorded container dimensions rather than the example renderer's current defaults.
for j,name in enumerate(['bed_floor','bed_wall_front','bed_wall_left','bed_wall_right','approach_bank'],2):
 body=getattr(cfg.scene,name);body.spawn.size=tuple(float(x) for x in 2*source['collider_half_size'][j]);body.spawn.visible=True
cfg.sim.physics.load_visual_shapes=True
cfg.sim.visualizer_cfgs=[v for v in cfg.sim.visualizer_cfgs if v.visualizer_type=='newton_rtx']
visual=cfg.sim.visualizer_cfgs[0];visual.headless=True;visual.origin_type='world';visual.origin_track_path=None
width,height=visual.window.size;top=60;size=(2*width,height+top)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
with launch_simulation(cfg.sim,dict(headless=True,device='cuda:0',visualizer=['newton_rtx'])):
 env=gym.make(task,cfg=cfg)
 try:
  model=NewtonManager.get_model();state=NewtonManager.get_state_0();visualizer=env.unwrapped.sim.visualizers[0]
  assert model.particle_count==meta['particle_count'] and model.body_count==snap['body_pose'].shape[1]
  np.testing.assert_allclose(model.body_com.numpy(),source['body_com_local'][:],atol=1e-6)
  # Both panels use the exact same reference body poses and their respective saved particle positions.
  for j in range(len(snap['time_s'])):
   frame=meta['start_frame']+j*4
   np.testing.assert_allclose(snap['body_pose'][j],source['body_pose'][frame],atol=1e-6)
   np.testing.assert_allclose(snap['reference_position'][j],source['particle_position'][frame],atol=1e-6)
  a.output.parent.mkdir(parents=True,exist_ok=True)
  writer=None
  if not a.preview:writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','50','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
  # Frames 0 through 249 cover exactly five seconds at 50 fps; the endpoint remains in the saved data.
  indices=[100] if a.preview else range(len(snap['time_s'])-1)
  for k in indices:
   t=float(snap['time_s'][k]);image=Image.new('RGB',size,'white');draw=ImageDraw.Draw(image)
   for i,(label,key) in enumerate([('Implicit MPM','reference_position'),('Particle NeRD','particle_position')]):
    state.body_q.assign(snap['body_pose'][k]);state.particle_q.assign(snap[key][k]);NewtonManager._mark_transforms_changed();NewtonManager._mark_particles_dirty()
    env.unwrapped.scene.update(.02);visualizer._camera_tracking_step=-1;visualizer.update_camera_tracking();visualizer._render_frame()
    pixels=visualizer._viewer.get_frame().numpy();image.paste(Image.fromarray(pixels).convert('RGB'),(i*width,top))
    draw.text((i*width+20,15),label,font=font,fill='#17365d' if i==0 else '#187a3b')
    draw.text((i*width+width-170,15),f't = {t:.2f} s',font=font,fill='#303030')
   if k==100:image.save(a.output.with_suffix('.jpg'),quality=90)
   if writer:writer.stdin.write(np.asarray(image).tobytes())
   if k%50==0:print('FRAME',k,flush=True)
  if writer:writer.stdin.close();assert writer.wait()==0
  print('VIDEO_DONE',a.output,flush=True)
 finally:env.close()
