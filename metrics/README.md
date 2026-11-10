# Particle NeRD evaluation metrics

This document defines metrics for comparing Particle NeRD with implicit MPM. Measured results are available in the [object-model benchmark report](https://jnskkmhr.github.io/Particle_NeRD_benchmark/benchmark_environment/object_prediction/): Figures 1–7 show the core accuracy metrics, and Figures 8–9 compare the fixed-object consistency tests. Particle trajectories, sand shape, material distribution, and contact forces describe different aspects of accuracy and should be reported together.

## Evaluation conditions

Use the same initial particle state, material parameters, interacting-body geometry, timestep, coordinate frame, and boundary conditions for each paired experiment. Record the checkpoint hash, code revision, particle count, particle spacing, active-particle selection settings, and all displacement stabilization or force-filtering settings. Evaluate the deployed method with its configured postprocessing; label any ablations separately.

Distinguish three evaluation modes:

1. **Force reconstruction with reference particles:** supply recorded MPM particle states and interacting-body motion to the force estimator. This isolates force reconstruction from neural particle-prediction error. Specify the particle-velocity definition and how proxy free velocities are obtained, including any use of recorded MPM forces.
2. **Particle rollout with prescribed body motion:** initialize the particle history from the reference and replay the same recorded body trajectory into both methods. Particle NeRD then advances its own particle state without replacing predictions with reference particles. This measures accumulated particle-prediction error under matched body motion. Report one-step predictions separately from these recurrent rollouts. Disclose any additional reference quantities supplied to force reconstruction.
3. **Closed-loop simulation:** let each coupled solver and policy evolve independently from matched initial conditions. Report locomotion, forces, and deformation, but do not interpret differences as isolated neural prediction error once the body trajectories diverge.

Compare states at the same physical times. Report error curves throughout the rollout, not only a final value. Include standing sequences of at least 20 seconds when collecting new reference data to expose delayed material loss and force drift. Do not extrapolate stability beyond the available recording.

## Core metric set

| Metric | Unit | What it measures |
|---|---|---|
| Particle-position RMSE, globally and by region | mm | Accuracy of corresponding particle trajectories |
| Mean and 95th-percentile symmetric surface distance | mm | Accuracy of the reconstructed sand shape |
| Material-mass discrepancy on a common grid | dimensionless; local differences in kg | Incorrect redistribution or loss of material |
| Velocity-field RMSE on a common grid | m/s | Accuracy of local material motion |
| Per-body force RMSE and signed mean error for Fx, Fy, Fz | N | Accuracy and bias of material support and friction forces |
| Spatio-temporal consistency under a fixed object | mm, mm/s; mass fractions | Persistent particle drift, shape change, and local material depletion |

Kinetic energy and point-cloud distribution distances are supplementary diagnostics. No single aggregate score replaces this set.

## 1. Particle-position error

Let x_i and y_i denote the MPM and Particle NeRD positions of the same particle at a given time. For a set A of corresponding particle IDs, define the three-dimensional position RMSE as

$$
E_x = \sqrt{\frac{1}{|A|}\sum_{i\in A}\|y_i-x_i\|_2^2}.
$$

This is the RMS Euclidean position error per particle, without an additional division by three. Convert meters to millimeters for reporting.

Report it for the entire bed and for two additional regions:

- **Near-contact particles:** reference particles within a specified distance of the prescribed interacting-body surfaces.
- **Moving particles:** reference particles whose speed exceeds a specified threshold, using a declared velocity definition.

Choose thresholds before comparing methods and report the number of particles in each region. Use the same reference-defined IDs for both methods at each time. Empty regions are reported as unavailable, not as zero error.

Position RMSE is meaningful when particle identities are preserved. It penalizes incorrect trajectories, including particle rearrangements that leave a similar bulk shape. A whole-bed average can also hide large contact-region errors among many nearly stationary particles. These are reasons to supplement RMSE, rather than discard it as noise.

For moving patches, recycled particle slots must not be treated as persistent material identities. Track material identity or restrict correspondence metrics to particles with verified shared identities. Otherwise use spatial field metrics.

## 2. Sand-surface error

Reconstruct a material-density field and an isosurface from each particle set using identical kernels, grid spacing, grid bounds, and density thresholds. Document these settings and check sensitivity to the reconstruction resolution.

Sample points uniformly by surface area on each mesh and measure their distances to the other mesh. Give the two directions equal weight when reporting the mean and 95th percentile of the pooled distance distribution. Report distances in millimeters and record the sampling count and seed. Comparing mesh vertices directly can bias the result toward densely triangulated regions.

Compare surfaces in the original shared coordinate frame. Do not apply rigid alignment, translation correction, or scale fitting: incorrect sand displacement or settling is part of the prediction error.

A signed-distance-field comparison is an optional alternative. Define the sign convention, sampling grid, and evaluation band around the union of the two surfaces; a whole-domain average can be dominated by unchanged space. If a surface is missing or reconstruction fails, record the failure rather than assigning zero error.

Surface metrics measure footprints, pile formation, and bulk deformation. Their ability to resolve cavities depends on reconstruction scale, so they must be accompanied by material-distribution measurements beneath the interacting bodies.

## 3. Material distribution and sub-foot depletion

Deposit physical particle mass onto a shared grid using the same normalized deposition weights. Let a_c and b_c be the MPM and Particle NeRD masses in cell c. With M equal to the common initial material mass, report

$$
E_m = \frac{\sum_c |b_c-a_c|}{M}.
$$

Also report total mass and local mass-difference maps. Use physical particle mass, not the force estimator's effective mass scales. Evaluate the complete represented material, including particles outside the neural active set. Grid bounds must cover both particle sets; material outside the evaluation domain must be accounted for explicitly.

For each G1 foot, additionally measure material mass in a prescribed region beneath its sole. Define a sole frame with z = 0 on the sole and positive z directed into the foot. The region uses the sole footprint in x and y and a declared interval -d <= z < 0, clipped to the material-accessible region above the rigid bed floor. Use the same prescribed foot pose and depth d for both methods, and record these definitions with the results.

Plot this mass in kilograms and its change from initialization over time. It directly measures the depletion that can precede loss of support. Particle counts may be included when particle masses and resolution match, but counts alone are not comparable across resolutions. Report material inside the collider volume separately to detect penetration.

### Supplementary point-cloud distances

**Chamfer distance** compares nearest-neighbor distances in both directions and does not require persistent particle identities. Specify whether distances are squared or unsquared and the associated units. It can miss incorrect local density because many particles can match the same nearby reference particle; it should not replace the grid-mass metric. See the [density-aware Chamfer study](https://arxiv.org/abs/2111.12702).

**Earth mover's distance / Wasserstein distance** measures transport between material distributions. Use physical mass weights and specify the transport cost and treatment of unequal total mass. Exact evaluation on full particle sets can be expensive. Sliced Wasserstein is a supplementary alternative; report the number of projection directions, random seed, and any sampling. Label approximations explicitly. See [Point-Set Distances for Learning Representations of 3D Point Clouds](https://openaccess.thecvf.com/content/ICCV2021/html/Nguyen_Point-Set_Distances_for_Learning_Representations_of_3D_Point_Clouds_ICCV_2021_paper.html).

## 4. Velocity fields and kinetic energy

Use a consistent particle-velocity definition. For example, derive velocity from consecutive recorded positions for both methods at the same timestep. Do not silently compare a finite-difference velocity with a differently timed native-solver velocity.

Deposit mass and momentum onto the same grid, then divide momentum by mass to obtain cell velocity. Let v_c and u_c be the reference and predicted velocities. For cells C with sufficient material mass in both methods, report the reference-mass-weighted error

$$
E_v = \sqrt{\frac{\sum_{c\in C} a_c\|u_c-v_c\|_2^2}{\sum_{c\in C} a_c}}.
$$

Declare the cell-mass threshold and report the fraction of reference mass covered by C. Missing material is assessed by the mass metric; it must not disappear silently from the evaluation when empty cells are excluded. If C is empty, report the velocity metric as unavailable.

For each method, also plot particle kinetic energy

$$
K = \frac{1}{2}\sum_i m_i\|v_i\|_2^2
$$

in joules, using physical particle masses and velocities in the same inertial frame. Regional energy curves near interacting bodies can reveal residual agitation or excessive damping that a global total hides.

Kinetic energy is a diagnostic, not a complete motion metric: different velocity directions can produce the same energy. Sand dissipates energy and exchanges work with the robot and boundaries. Do not use constant kinetic-plus-gravitational energy as a correctness criterion without accounting for those exchanges and internal material energy.

## 5. Contact-force accuracy and standing stability

Compare the force exerted by the material on each interacting body in the same world coordinate frame. For G1, report each foot separately with Fx, Fy, and Fz in newtons. Document the axis convention, force sampling interval, and whether rigid-floor contact is included; both signals must include the same force sources.

For each component, report time-series overlays, RMSE, and signed mean prediction error. Define force-vector RMSE as the square root of the mean squared Euclidean force error, without division by three, and report it separately from component RMSE. If comparing torques, use the same reference point, such as each body's center of mass, and report N m.

Standing tests should show force drift together with material mass beneath each sole. Low force variance alone does not imply correct support: a quiet prediction can still underestimate the sustained load. Report means and variability over declared time windows, and retain the full time series.

Use raw forces for the primary comparison. If force filtering is evaluated, document its parameters and initialization and show its delay and effect on the closed-loop behavior separately. Do not shift force traces in time to conceal prediction lag.

## Reporting and reproducibility

Compute summaries per trajectory before aggregating across trajectories, so long recordings do not receive accidental extra weight. Report trajectory counts, duration, evaluation mode, region definitions, and variability across trajectories. Distinguish frame-level percentiles from variability across independent runs.

Record all metric parameters, sampling seeds, and data/checkpoint hashes alongside results. Include failed rollouts and their failure times; do not silently omit them or average only surviving trajectories without disclosing the available duration and sample count.

Every figure must identify its metric, units, reference, and evaluation mode. Use “Figure X:” captions and explain each labeled panel. Any accompanying simulation video must show the same recorded rollout, checkpoint, time interval, and solver settings as the plotted results. Historical or differently conditioned results must be identified explicitly.


## 6. Spatio-temporal consistency under a stationary object

Hold the interacting body at a fixed world pose and verify its position, orientation, and linear/angular velocities during measurement. Track whether the sand continues to move despite the stationary body. This is a diagnostic of material evolution; a fixed object does not respond to loss of support.

For each method separately, choose a shared physical start time t0 and compare that method's later particles with its own state at t0. Do not replace neural predictions with MPM particles at t0. With physical particle masses m_i and total mass M, report

$$
D(t)=\sqrt{\frac{\sum_i m_i\|x_i(t)-x_i(t_0)\|^2}{M}},\qquad
Z(t)=\frac{\sum_i m_i[z_i(t)-z_i(t_0)]}{M}.
$$

D is RMS particle drift; Z is signed mean vertical drift, negative downward in the shared world frame. Report both in millimeters. Fit a slope to Z over a declared late interval and report mm/s to expose sustained sinking. This is an ordinary least-squares slope, not a smoothed force or position signal.

Also measure symmetric mean/p95 distance to the same method's t0 surface, temporal grid-mass change `sum_c |a_c(t)-a_c(t0)| / M`, physical mass in the fixed region beneath the object, kinetic energy, and mass with particle centers below the rigid tray floor. Report a second leakage threshold 1 cm beneath the floor to distinguish small boundary crossing from deeper escape. Preserve all particle identities and physical masses; do not align surfaces, discard outliers, or clip particles to the tray for evaluation.

The current experiment includes one box, sphere, and cylinder pair with particle seed 42. Object insertion runs from 1 to 2 s, followed by a fixed pose through 23 s. Measurement is 3–23 s; the late drift-rate interval is 13–23 s. Both methods evolve from the same initial particles at 0 s. The 3 s reference is a measurement anchor, not a claim that either bed has reached equilibrium. No training, active selection, displacement gate, particle projection, or force filtering was added for these experiments.

**Reference quality matters:** these MPM recordings also settle and show floor leakage. The box and cylinder each contain one deeply escaped MPM particle, which is retained and strongly affects RMS drift. The original paired-field grid hit its 20-million-node limit for those cases. Their paired field results remain unavailable. The separate temporal computation uses the same deposition and surface rules with an explicitly declared 100-million-node limit and all particles retained. Report these limitations with the measurements; neither numerical completion nor a quiet force establishes static material accuracy.

## Available measured results

| Metric | Measured population | Report location |
|---|---|---|
| Global, near-contact, and moving-particle position errors | 60 held-out, six-second trajectories | Figures 1–2; time curves on reference case pages |
| Mean and 95th-percentile symmetric surface distance | 60 held-out trajectories | Figure 3; reconstruction-resolution sensitivity in the downloadable `test_report.json` |
| Grid-mass discrepancy, beneath-object mass, penetration, and total represented mass | 60 held-out trajectories | Figure 4; maps and curves on reference pages |
| Velocity RMSE, reference-mass coverage, and kinetic energy | 60 held-out trajectories | Figure 5 |
| Fx/Fy/Fz and vector force errors, signed bias | Four dedicated reference recordings; no force references in the 60-test dataset | Figures 6–7; additional fixed-hold force traces on their pages |
| Spatio-temporal consistency | Three new fixed-object recordings, 20-second measurement interval | Figures 8–9 on the overview; Figure 7 time curves and exact matching video on each fixed-hold page |

All result plots use the same frozen object-model checkpoint from training update 9,000 (SHA-256 `e22f611d594449ba2eeb54ca7b21203e9ca856de82c8105de13f80190ef5ae0f`). This is the object benchmark model, not certification of a separately modified G1 runtime or another model checkpoint. Optional Chamfer/Wasserstein distances have not been measured. Raw datasets and model weights are private; public reports include plots, videos, scalar metrics, definitions, and provenance.
