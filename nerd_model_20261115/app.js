"use strict";
const $ = id => document.getElementById(id);
const f = (x, d = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
let CASES = [], current = null;
function row(a, b) { const tr = document.createElement("tr"); for (const t of [a, b]) { const td = document.createElement("td"); td.textContent = t; tr.append(td); } return tr; }
function statsRows(c) {
  const s = c.stats, rows = [];
  if (c.family === "object") {
    rows.push(["Object height at end, NeRD / MPM", `${f(s.z_nerd)} m / ${f(s.z_mpm)} m`]);
    rows.push(["RMS surface change near object, NeRD / MPM", `${s.rms_nerd.toFixed(1)} mm / ${s.rms_mpm.toFixed(1)} mm`]);
  } else {
    rows.push(["Centre of mass in x at end, NeRD / MPM", `${f(s.com_nerd, 2)} m / ${f(s.com_mpm, 2)} m`]);
    rows.push(["Mean horizontal spread at end, NeRD / MPM", `${s.spread_nerd.toFixed(3)} m / ${s.spread_mpm.toFixed(3)} m`]);
  }
  rows.push(["Surface distance, mean at end", `${s.dist_mean.toFixed(1)} mm`]);
  rows.push(["Surface distance, mean over time", `${s.dist_time.toFixed(1)} mm`]);
  rows.push(["Surface distance, 95th percentile at end", `${s.dist_p95.toFixed(1)} mm`]);
  rows.push(["Kinetic energy integrated over time, NeRD / MPM", `${s.ke_int_nerd.toFixed(3)} J·s / ${s.ke_int_mpm.toFixed(3)} J·s`]);
  rows.push(["Peak kinetic energy, NeRD / MPM", `${s.ke_peak_nerd.toFixed(3)} J / ${s.ke_peak_mpm.toFixed(3)} J`]);
  rows.push(["Duration", `${c.duration} s`]);
  return rows;
}
function setVideo(el, c, name) { el.poster = `${c.id}/${name}.jpg`; el.src = `${c.id}/${name}.mp4`; el.load(); }
function choose(id) {
  const c = CASES.find(x => x.id === id); if (!c) return; current = id; history.replaceState(null, "", "#" + id);
  $("category").textContent = c.family === "object" ? `${c.shape.toUpperCase()} · CLOSED LOOP` : `${c.shape.toUpperCase()} OF SAND · FLAT GROUND`;
  $("title").textContent = c.title; $("description").textContent = c.description;
  setVideo($("v-plot"), c, "plot"); setVideo($("v-dist"), c, "surface_distance"); $("v-ke").src = `${c.id}/kinetic_energy.png?v=2`;
  const hasSim = c.sim; $("sim-screen").hidden = !hasSim; $("v-sim").hidden = !hasSim;
  if (hasSim) setVideo($("v-sim"), c, "simulation"); else { $("v-sim").removeAttribute("src"); }
  $("sim-note").textContent = "Isaac Lab viewer recording. Left: Particle NeRD. Right: MPM.";
  $("plot-note").textContent = c.family === "object"
    ? "Left: Particle NeRD. Right: MPM. Top: side slab of particles, coloured by displacement from the start of the motion. Bottom: change in surface height, top view."
    : "Left: Particle NeRD. Right: MPM. Top: side view of all particles. Bottom: top view of all particles. Both coloured by z position.";
  $("stats").replaceChildren(...statsRows(c).map(([a, b]) => row(a, b)));
  for (const b of $("cases").children) b.setAttribute("aria-pressed", String(b.dataset.id === id));
}
function filter() {
  const fam = $("family").value, q = $("search").value.trim().toLowerCase();
  const vis = CASES.filter(c => (!fam || c.family === fam) && (!q || (c.title + " " + c.motion).toLowerCase().includes(q)));
  $("count").textContent = `${vis.length} of ${CASES.length} cases`; $("empty").hidden = vis.length > 0;
  $("cases").replaceChildren(...vis.map(c => { const b = document.createElement("button"); b.dataset.id = c.id; b.textContent = c.title; const s = document.createElement("small"); s.textContent = c.family === "object" ? "Closed loop" : "Flat ground"; b.append(s); b.onclick = () => choose(c.id); b.setAttribute("aria-pressed", String(c.id === current)); return b; }));
  if (vis.length && !vis.some(c => c.id === current)) choose(vis[0].id);
}
fetch("cases.json").then(r => r.json()).then(data => {
  CASES = data;
  for (const c of CASES) {
    const tr = document.createElement("tr"), s = c.stats;
    const cells = c.family === "object"
      ? [c.title, `${f(s.z_nerd)} / ${f(s.z_mpm)}`, `${s.rms_nerd.toFixed(1)} / ${s.rms_mpm.toFixed(1)}`, s.dist_mean.toFixed(1), s.dist_time.toFixed(1), `${s.ke_int_nerd.toFixed(3)} / ${s.ke_int_mpm.toFixed(3)}`]
      : [c.title + " · " + c.motion.split(" (")[0], `${f(s.com_nerd, 2)} / ${f(s.com_mpm, 2)}`, `${s.spread_nerd.toFixed(3)} / ${s.spread_mpm.toFixed(3)}`, s.dist_mean.toFixed(1), s.dist_time.toFixed(1), `${s.ke_int_nerd.toFixed(3)} / ${s.ke_int_mpm.toFixed(3)}`];
    for (const t of cells) { const td = document.createElement("td"); td.textContent = t; tr.append(td); }
    const td = tr.firstChild; const a = document.createElement("a"); a.href = "#" + c.id; a.textContent = td.textContent; a.onclick = e => { e.preventDefault(); $("family").value = ""; $("search").value = ""; filter(); choose(c.id); document.querySelector(".workspace").scrollIntoView({ behavior: "smooth" }); }; td.replaceChildren(a);
    (c.family === "object" ? $("summary") : $("summary-toss")).append(tr);
  }
  $("family").onchange = filter; $("search").oninput = filter;
  const want = location.hash.slice(1); current = CASES.some(c => c.id === want) ? want : null; filter();
  if (current) choose(current);
});
