We want to have 5-6 benchmark environments where it involves different object shapes and intrusion motions.
I am thinking about having elementary task like box pushing, circling etc like https://github.com/newton-physics/newton#mpm-examples
And HTML website layout can be like this: https://github.com/jnskkmhr/motion_library

What I would include is

Elementary object motion
* Box pushing horizontally
* Box pushing toward -z direciton
* Box circling
* Sphere rolling on surface
* Throwing flat shallow box in projectile motion (like how qudrupedal/humanoid) walks on sand

Different terrain example
Box/ball motion on
* Flat terrain
* Slope (down and up hill)
* Dune (wave terrain)

Robotics example
* Manipulator pushing sand with IK controller
* Quadurpedal robot walking on sand with IK controller
* Humanoid walking on sand with RL policy


For each table,
* can you include vide of simulation with MPM
* save each script to run simulation and generate dataset/benchmark results in /home/jkamohara/isaac/IsaacLab-particle-nerd/examples/particle_nerd_benchmark (implement using isaaclab, and use manager based)
    * you can define each task like
    ```
    /home/jkamohara/isaac/IsaacLab-particle-nerd/examples/particle_nerd_benchmark
    |-- core/
    |-- config/
        |-- cube
        |-- sphere
        |-- robot

    # follow mdp, config, env_cfg, task config instructions in /home/jkamohara/isaac/IsaacLab-particle-nerd/skills/env_construction.md
    # make each config stadalone as much as possible to improve readability
    # core can store shared code such as data collection, terrain configuration, data collection script/play script etc
    ```
* simulation can be ran 5-10second

## Implemented benchmark suite — 2026-11-09

The original plan above is implemented in
`examples/particle_nerd_benchmark`
under `/home/jkamohara/isaac/IsaacLab-particle-nerd`.

**Gallery:** http://localhost:8765/benchmark_environment/ (forward port 8765 when using SSH).
The gallery includes conditions, playback, matching force figures, dataset downloads,
and the reproduction command for each case.

| Motion | Recorded terrains | Cases |
| --- | --- | ---: |
| Horizontal box push | Flat, ascending, descending, wave | 4 |
| Vertical box intrusion | Flat | 1 |
| Box circling | Flat | 1 |
| Sphere with prescribed translation and spin | Flat, ascending, descending, wave | 4 |
| Dynamic shallow-box projectile | Flat | 1 |
| Franka pushing with end-effector IK | Flat | 1 |
| Freely balancing Go2 crawl with leg IK | Flat | 1 |
| G1 walking with an MPM-trained RL policy | Flat | 1 |

All **14 runs passed the recorded checks**. Each contains six seconds of native
implicit MPM simulation at 200 Hz, a 300-frame video at 50 fps, an HDF5 trajectory,
metadata with artifact hashes, a source snapshot, and an unsmoothed Fx/Fy/Fz plot.
The video and plot use the same rollout. These are MPM reference datasets; they do
not measure Particle NeRD prediction error or solver scalability.

The Go2 base is free: only its leg joints are actuated. Its successful run uses a
**0.12 m sand layer over a rigid floor**, which its feet may also contact; it advances
0.111 m in six seconds. This is not a deep-sand support test. The G1 uses a 0.30 m bed
and the MPM-trained policy `jqur5jhm`, iteration 19,999, with a 0.35 m/s command; it
advances 1.658 m. The policy checkpoint and saved training configuration are retained
in `checkpoints/jqur5jhm/`.

Reported forces are the native MPM impulse divided by the physics interval, summed
over the reacting bodies, in world axes. They exclude direct rigid-floor and wall
forces. The scripts save achieved robot motion; prescribed object motion is labeled
explicitly. HDF5 trajectories do not include the complete MPM constitutive history
required for restart.

Validation checks include finite recorded states and forces, final particle
containment above the tray floor, material contact, robot posture and forward
motion, and video frame/timestamp alignment. The gallery's filters, case navigation,
video playback, and mobile layout were also checked. These checks establish execution
and artifact consistency, not agreement with physical sand experiments.

See `catalog.json`, `validation_summary.json`, and each `runs/*/validation.json` for
the recorded cases and measurements. Source commands and simulation parameters are
in the implementation README linked above.


## Website ownership and publication

The HTML, CSS, JavaScript, and `publish.py` are maintained in this directory.
No website template is maintained inside the IsaacLab simulation example package.

- [Published gallery](https://jnskkmhr.github.io/Particle_NeRD_benchmark/benchmark_environment/)
- [GitHub repository](https://github.com/jnskkmhr/Particle_NeRD_benchmark)
- Datasets and checkpoint: retained locally; excluded from this publication.

Large HDF5 and checkpoint files remain local. `../local_artifacts.json` records
their local paths and SHA-256 checksums. Run `python publish.py --help` for
catalog regeneration; public raw-data links are omitted from this gallery.
