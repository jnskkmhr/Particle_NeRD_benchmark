"use strict";
const clips = JSON.parse(document.getElementById("motion-video-data").textContent);
const shape = document.getElementById("motion-shape");
const shapeWrap = document.getElementById("motion-shape-wrap");
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
const FREEFALL_MOTIONS = [
  ["drop", "Free-fall drop (flat ground)"],
];

function syncControls() {
  const isFreefall = modeSelect && modeSelect.value === "freefall";
  if (shape) {
    shape.disabled = isFreefall;
    if (shapeWrap) {
      shapeWrap.style.opacity = isFreefall ? "0.45" : "1.0";
    }
  }
  if (modeSelect) {
    const previousMotion = motion.value;
    const allowed =
      modeSelect.value === "prescribed"
        ? KINEMATIC_MOTIONS
        : modeSelect.value === "freefall"
          ? FREEFALL_MOTIONS
          : DYNAMIC_MOTIONS;
    motion.innerHTML = "";
    for (const [val, label] of allowed) {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = label;
      motion.appendChild(opt);
    }
    if (allowed.some(([val]) => val === previousMotion)) {
      motion.value = previousMotion;
    } else if (motion.options.length > 0) {
      motion.selectedIndex = 0;
      motion.value = motion.options[0].value;
    }
  }
  updateMotion();
}

function updateMotion() {
  const isFreefall = modeSelect && modeSelect.value === "freefall";
  const targetShape = isFreefall ? "freefall" : shape.value;
  const targetMode = isFreefall ? "dynamic" : (modeSelect ? modeSelect.value : "prescribed");
  let clip = clips.find(
    item =>
      item.shape === targetShape &&
      item.motion_mode === targetMode &&
      item.motion === motion.value
  );
  if (!clip) {
    clip = clips.find(item => item.shape === targetShape && item.motion_mode === targetMode) || clips[0];
  }
  if (!clip) return;
  if (motion.value !== clip.motion) {
    motion.value = clip.motion;
  }
  video.pause();
  video.poster = `motions/${clip.episode_id}/comparison_poster.png`;
  video.querySelector("source").src = `motions/${clip.episode_id}/comparison.mp4`;
  video.load();
  const modelLabel = clip.model_label || "Particle NeRD";
  const headerLabel =
    clip.shape === "freefall"
      ? "Sand block on flat ground [Dynamic: unconfined free-fall]"
      : `${shape.options[shape.selectedIndex].text} [${
          clip.motion_mode === "prescribed" ? "Kinematic: prescribed pose" : "Dynamic: gravity & contact response"
        }]`;
  document.getElementById("motion-description").textContent =
    `${headerLabel} — ${clip.motion_label.toLowerCase()}, episode ${clip.episode_id}. ` +
    `${clip.motion_description} Left: implicit MPM. Right: ${modelLabel}. ` +
    `Saved states at 50 fps, simulation time 0.02–${clip.duration_s || 6} s.`;
}

if (shape) {
  shape.addEventListener("change", () => syncControls());
}
if (modeSelect) {
  modeSelect.addEventListener("change", () => syncControls());
}
motion.addEventListener("change", updateMotion);
syncControls();
