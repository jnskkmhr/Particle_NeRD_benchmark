# G1 foot-force estimation

[Open the current report](index.html). It evaluates the currently installed Particle NeRD solver on one five-second recorded G1 trajectory against implicit MPM. Argon’s foot-speed displacement scaling is enabled. The plots and 50 fps video use the same saved run.

Force-vector RMSE is **83.60 N**, averaged over both feet and 1,000 steps at 200 Hz. Forces are expressed in the world frame and measured in newtons. The robot follows prescribed recorded motion; the predicted forces do not drive the robot.

See [run metadata](../data/current-20261109/metadata.json), [alignment verification](../data/current-20261109/verification.json), and [reproduction instructions](../../plots/CURRENT_REPLAY.md).

`validation-force.json` is a separate historical October 28 archive, not the data displayed in the current report.
