
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

**TODO**

* storing `ee_poses` in `CapVert` is not necessary, should think of a way to get around this.
* storing joint data in a continuous array may not be necessary, since we are using nested list to describe subprocesses anyway
* need to regulate the use of `ik_joints` or `ik_joint_names` for user interfaces

Unreleased
----------

**Added**

0.2.0
----------

**Added**

* `SparseLadderGraph` completed
* export planned trajectory for extrusion
* add parsing function for visualizing saved extrusion trajectories
* `from_data` methods for `Trajectory` and subclasses
* tagging print processes with `ground`/`creation`/`connect` in the test function
* infinite pose sampler added for extrusion case when using sparse ladder graph to solve
* Added `max_valence_extrusion_direction_routing` to `extrusion.utils`
* Added `reverse_flags` info to `add_collision_fns_from_seq` and extrusion's test
* Added `start_conf` parameter to `SparseLadderGraph.extract_solution` and `solve_ladder_graph_from_cartesian_process_list` to allow minimizing ladder graph with respect to a given start configuration
* Added `picknplace.transition_planner`
* Added `target_conf` attribute to `CartesianProcess` to allow using `snap_sols` when `sample_ik_sols` is called. This is essential for robots with large joint limits, e.g. UR.

**Minor**

* `is_any_empty` utility function for checking ik sol list of lists
* `reset_ee_pose_gen_fn` for easier resetting generator
* Added print_table model in the `mit_3-412_workspace` URDF/SRDF

**Removed**

* Removed `PicknPlaceBufferTrajectory`'s `ee_attachments` and `attachments` attributes
* Removed `picknplace.planner_interface` (which is there only as an archive)

**Fixed**

* fix nested empty list detection bug in `is_any_empty`
* add `disabled_collisions` argument to the extrusion transition_planner
* Fixed `min_z` to `base_point` model transformation in `extrusion.parsing`

**Changed**

* extrusion export save `lin_path`'s poses as 4x4 tform matrix (there's some disagreement in quaterion in `compas.Frame.from_quat`?)
* move extrusion test fixtures into a separate fixture module
* ladder graph interface broken into `from_cartesian_process_list`, `from_cartesian_process`, `from_poses` to increase code reuse
* Changed `sub_process_ids` specification in `prune_ee_feasible_directions`
* Changed `Trajectory` to have `ee_attachments` and `attachments` attributes natively
* Changed `Trajectory`'s `from_data`, making it raise `ValueError` when robot body cannot be found in pybullet
* Changed `MoveTrajectory` to have `element_id` attributes natively
* Changed `picknplace.visualization` to reload and manually assign pybullet bodies to ensure objects get matched correctly
* Changed `build_picknplace_cartesian_process_seq` to inject `ee_attach` info before passing into ladder graph solver, and tag element attachment after solving is finished.


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

0.0.1
------

**Added**

* Initial version
