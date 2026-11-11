const $=id=>document.getElementById(id);let selected=null;let visible=[];
const terrainNames={flat:'Flat sand bed',ascend:'Ascending slope (+12%)',descend:'Descending slope (−12%)',wave:'Wave (0.06 m amplitude, 1.2 m wavelength)'};
const sourceRepository='https://github.com/jnskkmhr/Particle_NeRD_benchmark';
const publicGallery='https://jnskkmhr.github.io/Particle_NeRD_benchmark/benchmark_environment/index.html';
// Each case links to its own recorded source, not a potentially changed simulator checkout.
function environmentFiles(c){
  const robot=c.family==='robot';
  const family=robot?'robot':c.motion==='rolling'?'sphere':'cube';
  const name=robot?c.motion:family;
  const prefix=['g1','quadruped'].includes(c.motion)?c.motion+'_':'';
  const root='config/'+family;
  const files=[
    ['Environment configuration',root+'/'+name+'_env_cfg.py'],
    ['Scene and assets',root+'/env_cfg/'+prefix+'scene_cfg.py'],
    ['Actions',root+'/env_cfg/'+prefix+'action_cfg.py'],
    ['Observations',root+'/env_cfg/'+(c.motion==='g1'?'g1_':'')+'observation_cfg.py'],
    ['Reset events',root+'/env_cfg/event_cfg.py'],
    ['Physics configuration',prefix?'core/physics.py':root+'/env_cfg/physics_cfg.py'],
    ['Task registration',root+'/__init__.py'],
    ['Motion and terrain selection','core/case_config.py'],
    ['Sand and tray geometry','core/terrain.py'],
    ['Playback entry point','core/play.py'],
    ['Controller and dataset collection','core/run.py'],
    ['Recording implementation','core/recording.py'],
  ];
  if(!robot)files.push(['Object motion targets','core/motions.py'],['Object action implementation','core/mdp/actions.py']);
  if(c.motion==='quadruped')files.push(['Leg IK controller','core/quadruped_controller.py']);
  return files;
}
function sourceUrl(c,path='',directory=false){
  return sourceRepository+'/'+(directory?'tree':'blob')+'/main/benchmark_environment/'+c.path+'/source/'+path;
}
function sourceLink(c,path){
  const a=document.createElement('a'),code=document.createElement('code');
  a.href=sourceUrl(c,path);a.target='_blank';a.rel='noopener';code.textContent=path;a.append(code);return a;
}
function showSources(c){
  const files=environmentFiles(c),entry=files[0][1];
  $('environment-file').href=sourceUrl(c,entry);
  $('environment-file').textContent='examples/particle_nerd_benchmark/'+entry;
  $('source-variant').textContent='Selected case: motion = "'+c.motion+'", terrain = "'+c.terrain+'". These settings are applied in core/case_config.py.';
  $('source-folder').href=sourceUrl(c,'',true);
  $('share-case').href=publicGallery+'#'+c.id;
  $('source-files').replaceChildren(...files.slice(1).map(([label,path])=>{
    const tr=document.createElement('tr'),name=document.createElement('td'),file=document.createElement('td');
    name.textContent=label;file.append(sourceLink(c,path));tr.append(name,file);return tr;
  }));
}
function choose(id){const c=visible.find(v=>v.id===id)||visible[0];if(!c)return;selected=c;history.replaceState(null,'','#'+c.id);
showSources(c);
$('title').textContent=c.title;$('category').textContent=c.family==='robot'?'ROBOT CONTACTS':'ELEMENTARY OBJECT MOTION';$('description').textContent=c.description;
const version='?v='+c.artifacts['simulation.mp4'].slice(0,12);
$('video').src=c.path+'/simulation.mp4'+version;$('video').poster=c.path+'/poster.jpg'+version;$('video').load();$('duration').textContent=c.duration+' s · 50 fps';
$('video-caption').textContent='Video 1: '+c.title+' on '+terrainNames[c.terrain].toLowerCase()+'. Run '+c.id+'. Captured every fourth 200 Hz physics step.';
const rows=[['Terrain',terrainNames[c.terrain]],['Controller',c.controller],['Particles',c.particles.toLocaleString()],['Initial position jitter','Up to '+(c.particle_initialization.max_offset_per_axis_m*1000).toFixed(1)+' mm per axis; seed '+c.particle_initialization.seed],['Tray wall top','z = '+c.wall_top_z_m.toFixed(2)+' m relative to the tray origin'],['Bed dimensions',c.bed.map(x=>x+' m').join(' × ')],['Physics / control', '200 Hz / '+c.control_hz+' Hz'],['MPM solve','APIC; Q1 velocity; PIC27 collider; 25 iterations; tolerance 10⁻⁴'],['Native settling',c.family==='object'?'1 s before the commanded interaction':'Robot and material evolve together from reset'],['Policy',c.policy||'No learned policy'],['Validation', 'Recorded-state checks passed; '+c.date]];
$('conditions').replaceChildren(...rows.map(([k,v])=>{const tr=document.createElement('tr');for(const text of [k,v]){const td=document.createElement('td');td.textContent=text;tr.append(td)}return tr}));
for(const [id,file] of [['metadata','metadata.json'],['validation','validation.json']])$(id).href=c.path+'/'+file+version;
const local=['localhost','127.0.0.1','::1'].includes(location.hostname);const dataUrl=c.dataset_url||(local?c.path+'/trajectory.h5':null);
$('dataset').textContent=dataUrl?'Dataset (HDF5) ↓':'Dataset retained locally';$('dataset').setAttribute('aria-disabled',String(!dataUrl));if(dataUrl)$('dataset').href=dataUrl;else $('dataset').removeAttribute('href');
$('plot').src=c.path+'/forces.svg'+version;$('plot-link').href=c.path+'/forces.svg'+version;$('command').textContent=c.command;
[...$('cases').children].forEach(b=>b.setAttribute('aria-pressed',b.dataset.id===c.id));}
function filter(){visible=BENCHMARK_CASES.filter(c=>(!$('family').value||c.family===$('family').value)&&(!$('terrain').value||c.terrain===$('terrain').value)&&(c.title+' '+c.terrain).toLowerCase().includes($('search').value.toLowerCase()));
$('cases').replaceChildren(...visible.map(c=>{const b=document.createElement('button');b.dataset.id=c.id;b.textContent=c.title;const small=document.createElement('small');small.textContent=terrainNames[c.terrain];b.append(small);b.onclick=()=>choose(c.id);return b}));
$('count').textContent=visible.length+' recorded cases';$('empty').hidden=visible.length>0;document.querySelector('.detail').hidden=visible.length===0;
choose(selected?.id||decodeURIComponent(location.hash.slice(1)));}
for(const id of ['family','terrain','search'])$(id).addEventListener('input',filter);
$('previous').onclick=()=>choose(visible[(visible.indexOf(selected)-1+visible.length)%visible.length].id);$('next').onclick=()=>choose(visible[(visible.indexOf(selected)+1)%visible.length].id);
for(const c of BENCHMARK_CASES){const tr=document.createElement('tr');for(const text of [c.title,terrainNames[c.terrain]]){const td=document.createElement('td');td.textContent=text;tr.append(td)}const source=document.createElement('td');source.append(sourceLink(c,environmentFiles(c)[0][1]));tr.append(source);for(const text of [c.controller,c.duration+' s']){const td=document.createElement('td');td.textContent=text;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href='#'+c.id;a.textContent='Inspect run →';a.onclick=e=>{e.preventDefault();$('family').value='';$('terrain').value='';$('search').value='';filter();choose(c.id);document.querySelector('.workspace').scrollIntoView({behavior:'smooth'})};td.append(a);tr.append(td);$('summary').append(tr)}filter();
