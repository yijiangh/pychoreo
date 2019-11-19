
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

**Added**

* export planned trajectory for extrusion
* add parsing function for visualizing saved extrusion trajectories
* `from_data` methods for `Trajectory` and subclasses
* tagging print processes with `ground`/`creation`/`connect` in the test function

**Changed**

* extrusion export save `lin_path`'s poses as 4x4 tform matrix (there's some disagreement in quaterion in `compas.Frame.from_quat`?)
* move extrusion test fixtures into a separate fixture module

0.1.1
----------

**Added**

* cartesian process class for modeling general linear movement in the workspace
* ladder graph interface using the Cartesian process class
* `Trajectory` class for modeling result trajectory in different contexts (inherited classes)
* `display_trajectories` for extrusion
* some simple exceptions added for `LadderGraph` and `DAGSearch`
* subprocess modeling to have a more detailed control over Cartesian process modeling
* add `exhaust_iter` method to `CartisianProcess` which resets the generator
* add template class `GenFn` for generating functions
* add `PrintBufferTrajectory` to model approach/retreat trajectories

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
