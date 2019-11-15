
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

**Added**

* cartesian process class for modeling general linear movement in the workspace
* ladder graph interface using the Cartesian process class
* `Trajectory` class for modeling result trajectory in different contexts (inherited classes)
* `display_trajectories` for extrusion
* `Command` class, useful for modeling and collision recording, not in used now
* some simple exceptions added for `LadderGraph` and `DAGSearch`

**Changed**

* move transition planning to application context.
* conform to the latest `pybullet_planning`

**Removed**

* `assembly_datastructure`
* the old `extrusion.run` module, moved to the test file

**Fixed**

**Deprecated**

**TODO**

- need to regulate the use of `ik_joints` or `ik_joint_names` for user interfaces

0.0.1
------

**Added**

* Initial version
