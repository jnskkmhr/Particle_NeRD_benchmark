"use strict";
const clips = JSON.parse(document.getElementById("motion-video-data").textContent);
const shape = document.getElementById("motion-shape");
const motion = document.getElementById("motion-kind");
const video = document.getElementById("motion-video");
function updateMotion() {
  const clip = clips.find(item => item.shape === shape.value && item.motion === motion.value);
  video.pause();
  video.poster = `motions/${clip.episode_id}/comparison_poster.png`;
  video.querySelector("source").src = `motions/${clip.episode_id}/comparison.mp4`;
  video.load();
  const modelLabel = clip.model_label || "Particle NeRD";
  document.getElementById("motion-description").textContent =
    `${shape.options[shape.selectedIndex].text} — ${clip.motion_label.toLowerCase()}, episode ${clip.episode_id}. ` +
    `${clip.motion_description} Left: implicit MPM. Right: ${modelLabel}. ` +
    `Saved states at 50 fps, simulation time 0.02–6 s. This is one of ${clip.group_trajectory_count} plotted trajectories in this shape/motion group.`;
  document.getElementById("motion-provenance").href = `motions/${clip.episode_id}/video_provenance.json`;
}
shape.addEventListener("change", updateMotion);
motion.addEventListener("change", updateMotion);
