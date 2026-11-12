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
    const previousMotion = motion.value;
    const allowed = modeSelect.value === "prescribed" ? KINEMATIC_MOTIONS : DYNAMIC_MOTIONS;
    const availSet = new Set(
      clips.filter(c => c.shape === shape.value && c.motion_mode === modeSelect.value).map(c => c.motion)
    );
    const filtered = allowed.filter(([val]) => availSet.size === 0 || availSet.has(val));
    const list = filtered.length ? filtered : allowed;
    motion.innerHTML = "";
    for (const [val, label] of list) {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = label;
      motion.appendChild(opt);
    }
    if (list.some(([val]) => val === previousMotion)) {
      motion.value = previousMotion;
    } else if (motion.options.length > 0) {
      motion.selectedIndex = 0;
      motion.value = motion.options[0].value;
    }
  }
  updateMotion();
}

function updateMotion() {
  let clip = clips.find(
    item =>
      item.shape === shape.value &&
      (!modeSelect || item.motion_mode === modeSelect.value) &&
      item.motion === motion.value
  );
  if (!clip && modeSelect) {
    clip = clips.find(item => item.shape === shape.value && item.motion_mode === modeSelect.value);
  }
  if (!clip) {
    clip = clips.find(item => item.shape === shape.value) || clips[0];
  }
  if (!clip) return;
  if (modeSelect && modeSelect.value !== clip.motion_mode) {
    modeSelect.value = clip.motion_mode;
  }
  if (motion.value !== clip.motion) {
    motion.value = clip.motion;
  }
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
syncControls(modeSelect || shape);
