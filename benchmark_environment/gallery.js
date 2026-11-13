const $=id=>document.getElementById(id);let selected=null;let visible=[];
const terrainNames={flat:'Flat sand bed',ascend:'Ascending slope (+12%)',descend:'Descending slope (−12%)',wave:'Wave (0.06 m amplitude, 1.2 m wavelength)'};
const defaultPlotCaption='Figure 1: Sand-on-collider forces from the video’s recorded run. (a) Fx; (b) Fy; (c) Fz. For robot cases, forces are summed over the reacting robot links listed in the validation file. Direct rigid tray contact is excluded.';
const freefallPlotCaption='Figure 1: Free-fall kinematics and floor reaction from the video’s recorded run. (a) Vertical height above floor z_COM and z_95; (b) radial runout r_95 and r_50; (c) floor normal reaction Fz against static weight Mg.';
function choose(id){const norm=s=>String(s||'').replace(/-/g,'_');const c=visible.find(v=>v.id===id||norm(v.id)===norm(id))||visible[0];if(!c)return;selected=c;history.replaceState(null,'','#'+c.id);
$('title').textContent=c.title;$('category').textContent=c.family==='robot'?'ROBOT CONTACTS':(c.family==='freefall'||c.motion==='freefall'?'PARTICLE FREE-FALL':'ELEMENTARY OBJECT MOTION');$('description').textContent=c.description;
const videoDigest=c.artifacts['simulation.mp4']||c.artifacts['video.mp4'];const version='?v='+videoDigest.slice(0,12);
const terrainLabel=c.motion==='freefall'?'Flat ground (z = −0.30 m, open boundary)':terrainNames[c.terrain];
$('video').src=c.path+'/simulation.mp4'+version;$('video').poster=c.path+'/poster.jpg'+version;$('video').load();$('duration').textContent=c.duration+' s · 50 fps';
$('video-caption').textContent='Video 1: '+c.title+' on '+terrainLabel.toLowerCase()+'. Run '+c.id+'. Captured every fourth 200 Hz physics step.';
const wallDesc=c.wall_top_z_m==null?'Open flat floor at z = −0.30 m (no retaining walls)':'z = '+c.wall_top_z_m.toFixed(2)+' m relative to the tray origin';
const bedLabel=c.motion==='freefall'?'Initial sand block':'Bed dimensions';
const settleDesc=c.motion==='freefall'?'Released in free-fall under gravity from t = 0 s':(c.family==='object'?'1 s before the commanded interaction':'Robot and material evolve together from reset');
const rows=[['Terrain',terrainLabel],['Controller',c.controller],['Particles',c.particles.toLocaleString()],['Initial position jitter','Up to '+(c.particle_initialization.max_offset_per_axis_m*1000).toFixed(1)+' mm per axis; seed '+c.particle_initialization.seed],['Tray wall top',wallDesc],[bedLabel,c.bed.map(x=>x+' m').join(' × ')],['Physics / control', '200 Hz / '+c.control_hz+' Hz'],['MPM solve','APIC; Q1 velocity; PIC27 collider; 25 iterations; tolerance 10⁻⁴'],['Native settling',settleDesc],['Policy',c.policy||'No learned policy'],['Validation', 'Recorded-state checks passed; '+c.date]];
$('conditions').replaceChildren(...rows.map(([k,v])=>{const tr=document.createElement('tr');for(const text of [k,v]){const td=document.createElement('td');td.textContent=text;tr.append(td)}return tr}));
for(const [id,file] of [['metadata','metadata.json'],['validation','validation.json']])$(id).href=c.path+'/'+file+version;
const local=['localhost','127.0.0.1','::1'].includes(location.hostname);const dataUrl=c.dataset_url||(local?c.path+'/trajectory.h5':null);
$('dataset').textContent=dataUrl?'Dataset (HDF5) ↓':'Dataset retained locally';$('dataset').setAttribute('aria-disabled',String(!dataUrl));if(dataUrl)$('dataset').href=dataUrl;else $('dataset').removeAttribute('href');
$('plot').src=c.path+'/forces.svg'+version;$('plot-link').href=c.path+'/forces.svg'+version;$('command').textContent=c.command;
const figCap=document.querySelector('.result-figure figcaption');if(figCap)figCap.textContent=c.motion==='freefall'?freefallPlotCaption:defaultPlotCaption;
[...$('cases').children].forEach(b=>b.setAttribute('aria-pressed',b.dataset.id===c.id));}
function filter(){visible=BENCHMARK_CASES.filter(c=>(!$('family').value||c.family===$('family').value)&&(!$('terrain').value||c.terrain===$('terrain').value)&&(c.title+' '+c.terrain+' '+c.motion+' '+c.controller).toLowerCase().includes($('search').value.toLowerCase()));
$('cases').replaceChildren(...visible.map(c=>{const b=document.createElement('button');b.dataset.id=c.id;b.textContent=c.title;const small=document.createElement('small');small.textContent=c.motion==='freefall'?'Flat ground (open boundary)':terrainNames[c.terrain];b.append(small);b.onclick=()=>choose(c.id);return b}));
$('count').textContent=visible.length+' recorded cases';$('empty').hidden=visible.length>0;document.querySelector('.detail').hidden=visible.length===0;
choose(selected?.id||decodeURIComponent(location.hash.slice(1)));}
for(const id of ['family','terrain','search'])$(id).addEventListener('input',filter);
$('previous').onclick=()=>choose(visible[(visible.indexOf(selected)-1+visible.length)%visible.length].id);$('next').onclick=()=>choose(visible[(visible.indexOf(selected)+1)%visible.length].id);
for(const c of BENCHMARK_CASES){const tr=document.createElement('tr');const terr=c.motion==='freefall'?'Flat ground (open boundary)':terrainNames[c.terrain];for(const text of [c.title,terr,c.controller,c.duration+' s']){const td=document.createElement('td');td.textContent=text;tr.append(td)}const td=document.createElement('td'),a=document.createElement('a');a.href='#'+c.id;a.textContent='Inspect run →';a.onclick=e=>{e.preventDefault();$('family').value='';$('terrain').value='';$('search').value='';filter();choose(c.id);document.querySelector('.workspace').scrollIntoView({behavior:'smooth'})};td.append(a);tr.append(td);$('summary').append(tr)}filter();
