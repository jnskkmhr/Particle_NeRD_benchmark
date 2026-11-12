"use strict";
const clips = JSON.parse(document.getElementById("motion-video-data").textContent);
const shape = document.getElementById("motion-shape");
const modeSelect = document.getElementById("motion-mode");
const motion = document.getElementById("motion-kind");
const video = document.getElementById("motion-video");

const KINEMATIC_MOTIONS = [
  ["horizontal", "Horizontal motion"],
  ["vertical", "Vertical motion"],
  ["circle", "Circular motion"],
  ["hold", "Holding"],
];
const DYNAMIC_MOTIONS = [
  ["drop", "Drop"],
  ["slide", "Slide"],
  ["throw", "Throw"],
  ["tumble", "Tumble"],
];

function syncControls(trigger) {
  if (shape.value === "freefall" && modeSelect) {
    if (trigger === modeSelect && modeSelect.value === "prescribed") {
      shape.value = "box";
    } else {
      modeSelect.value = "dynamic";
    }
  }
  if (modeSelect) {
    const currentMotion = motion.value;
    const allowed = modeSelect.value === "prescribed" ? KINEMATIC_MOTIONS : DYNAMIC_MOTIONS;
    const availSet = new Set(
      clips.filter(c => c.shape === shape.value && c.motion_mode === modeSelect.value).map(c => c.motion)
    );
    const filtered = allowed.filter(([val]) => availSet.size === 0 || availSet.has(val));
    motion.innerHTML = "";
    for (const [val, label] of (filtered.length ? filtered : allowed)) {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = label;
      motion.appendChild(opt);
    }
    if ([...motion.options].some(o => o.value === currentMotion)) {
      motion.value = currentMotion;
    }
  }
  updateMotion();
}

function updateMotion() {
  let clip = clips.find(item => item.shape === shape.value && item.motion === motion.value);
  if (!clip) {
    clip = clips.find(item => item.shape === shape.value) || clips[0];
    if (clip) {
      if (modeSelect) modeSelect.value = clip.motion_mode;
      motion.value = clip.motion;
    }
  }
  if (!clip) return;
  video.pause();
  video.poster = `motions/${clip.episode_id}/comparison_poster.png`;
  video.querySelector("source").src = `motions/${clip.episode_id}/comparison.mp4`;
  video.load();
  const modelLabel = clip.model_label || "Particle NeRD";
  const modeTag =
    clip.motion_mode === "prescribed"
      ? "[Kinematic: prescribed pose]"
      : "[Dynamic: gravity & contact response]";
  document.getElementById("motion-description").textContent =
    `${shape.options[shape.selectedIndex].text} ${modeTag} — ${clip.motion_label.toLowerCase()}, episode ${clip.episode_id}. ` +
    `${clip.motion_description} Left: implicit MPM. Right: ${modelLabel}. ` +
    `Saved states at 50 fps, simulation time 0.02–${clip.duration_s || 6} s. This is one of ${clip.group_trajectory_count} plotted trajectories in this group.`;
  document.getElementById("motion-provenance").href = `motions/${clip.episode_id}/video_provenance.json`;
}

shape.addEventListener("change", () => syncControls(shape));
if (modeSelect) {
  modeSelect.addEventListener("change", () => syncControls(modeSelect));
}
motion.addEventListener("change", updateMotion);
if (modeSelect) {
  syncControls(modeSelect);
}
