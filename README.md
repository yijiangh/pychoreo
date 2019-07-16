# pychoreo
<img src="docs/images/choreo_logo.png" alt="drawing" width="100"/>

choreo + pybullet

:construction: work in progress!

:pushpin: In the summer of 2019, *pychoreo* will be integrated into the [compas_fab](https://github.com/compas-dev/compas_fab) infrastructure. Stay tuned! :beers:

With *pychoreo*, you will be able to print the following cool structures (and many more!) with ease:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Vv7dEB8T_Jg" target="_blank"><img src="http://img.youtube.com/vi/Vv7dEB8T_Jg/0.jpg" alt="voronoi_extrusion"/></a>

The ROS implementation of choreo can be found [here](https://github.com/yijiangh/choreo).

[<img src="http://digitalstructures.mit.edu/theme/digistruct/images/digital-structures-logo-gray.svg" width="150">](http://digitalstructures.mit.edu/)
&nbsp; &nbsp; &nbsp; &nbsp;
[<img src="http://web.mit.edu/files/images/homepage/default/mit_logo.gif" width="80">](http://web.mit.edu/)

## <img align="center" height="15" src="https://i.imgur.com/x1morBF.png"/> Installation

```
$ git clone --recursive https://github.com/yijiangh/pychoreo.git
```

`pychoreo` has been tested again python `2.x` & `3.7`.

### Install pybullet

Install PyBullet on OS X or Linux using:

```
pip install numpy pybullet
```

### Install ikfast modules

To build ikfast modules:
```bash
$ cd conrob_pybullet/utils/ikfast/kuka_kr6_r900/
$ python setup.py build
```

For other robots, replace `kuka_kr6_r900` with the following supported robots:
- `eth_rfl`
- `abb_irb6600_track`

### Install conmech module for checking stiffness constraint

See [conmech](https://github.com/yijiangh/conmech) for more instructions.

### (optional) Install meshcat-python for sequence result visualization (WebGL-based)

See [meshcat-python](https://github.com/rdeits/meshcat-python) for instructions on
installing `meshcat` and `ZMQ`.

### Installation test

1. Test pybullet: `python -c 'import pybullet'`

2. Testing new IKFast modules

* `$ python -m conrob_pybullet.debug_examples.test_eth_rfl_pick`
* `$ python -m conrob_pybullet.debug_examples.test_irb6600_track_pick`

## <img align="center" height="15" src="https://i.imgur.com/x1morBF.png"/> Examples

### Extrusion

* `$ python -m choreo.extrusion.run`

`python -m choreo.extrusion.run -h` to see optional arguments.

After computation, you should be able to see result like the following:


<a href="http://www.youtube.com/watch?feature=player_embedded&v=q1S-7KQo1XU" target="_blank"><img src="http://img.youtube.com/vi/q1S-7KQo1XU/0.jpg" alt="simple_frame_demo"/></a>

To play with other examples, e.g. the topology optimized vault in [this paper](http://web.mit.edu/yijiangh/www//papers/HuangCarstensenMueller_IASS2018.pdf), try:
* `python -m choreo.extrusion.run -p topopt-100`

(in this example, the sequence planner will take about 150 sec to find a solution. Then, be patient with the transition planner...)

**Note**: If the terminal says something like:
```
transition planning # <> - E#<>
---
Warning: initial configuration is in collision
start extrusion pose:
pairwise BODY collision: body kr6_r900_workspace1 - body body<>
Press enter to continue
```

This problem is related to the end effector colliding with the element that it is currently printing. See [issue #2](https://github.com/yijiangh/pychoreo/issues/2). This is not hard to fix, once I get some time...

For now, simply type "Enter" to skip the current transition process and the planner will continue to the next one. If you keep skipping them (hopefully there are not a lot of them), you will get to see the following "result" with "magic jumps" between some extrusions (around 0:50):

<a href="http://www.youtube.com/watch?feature=player_embedded&v=mzu-OMvFMcE" target="_blank"><img src="http://img.youtube.com/vi/mzu-OMvFMcE/0.jpg" alt="simple_frame_demo_EE_self_collision"/></a>

### Pick-and-place

Coming soon...

## <img align="center" height="15" src="https://i.imgur.com/dHQx91Q.png"/> Citation

If you use this work, please consider citing as follows:

    @article{huang2018automated,
      title={Automated sequence and motion planning for robotic spatial extrusion of 3D trusses},
      author={Huang, Yijiang and Garrett, Caelan R and Mueller, Caitlin T},
      journal={Construction Robotics},
      volume={2},
      number={1-4},
      pages={15--39},
      year={2018},
      publisher={Springer}}

Algorithms behind Choreo:
- Automated sequence and motion planning for robotic spatial extrusion of 3D trusses, Constr Robot (2018) 2:15-39, [Arxiv-1810.00998](https://arxiv.org/abs/1810.00998)

Applications of Choreo:
- Robotic extrusion of architectural structures with nonstandard topology, RobArch 2018, [paper link](http://web.mit.edu/yijiangh/www/papers/Huang2019_RobArch.pdf)
- Spatial extrusion of Topology Optimized 3D Trusses, IASS 2018, [paper link](http://web.mit.edu/yijiangh/www//papers/HuangCarstensenMueller_IASS2018.pdf)

## <img align="center" height="15" src="https://i.imgur.com/H4NwgMg.png"/> Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/yijiangh/pychoreo/issues).

## <img align="center" height="15" src="https://i.imgur.com/x1morBF.png"/> Related repos<a name="related_repos"></a>

Task and Motion Planning
- https://github.com/caelan/pb-construction
- https://github.com/caelan/pddlstream

Computational design and digital fabrication
- https://github.com/compas-dev/compas_fab
