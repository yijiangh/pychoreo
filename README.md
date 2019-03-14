# pychoreo 
choreo + pybullet. 

The ROS implementation of choreo can be found [here](https://github.com/yijiangh/choreo).

## Installation

```
$ git clone --recursive https://github.com/yijiangh/pychoreo.git 
```

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

### Installation test

1. Test pybullet: `python -c 'import pybullet'`

2. Testing new IKFast modules

* `$ python -m conrob_pybullet.debug_examples.test_eth_rfl_pick`
* `$ python -m conrob_pybullet.debug_examples.test_irb6600_track_pick`

## Examples

* `$ python -m choreo.run`

## Citation

If you use this work, please consider citing the following work:

  @article{huang2018automated,
  title={Automated sequence and motion planning for robotic spatial extrusion of 3D trusses},
  author={Huang, Yijiang and Garrett, Caelan R and Mueller, Caitlin T},
  journal={Construction Robotics},
  pages={15--39},
  year={2018},
  publisher={Springer}}

Or [arxiv:1810.00998](https://arxiv.org/abs/1810.00998).
