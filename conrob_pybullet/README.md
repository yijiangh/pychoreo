# conrob_pybullet
construction robots' urdf, srdf models + ikfast modules for pybullet

## Installation

```
$ git clone https://github.com/yijiangh/conrob_pybullet.git
```

### Install pybullet

Install PyBullet on OS X or Linux using:

```
pip install numpy pybullet
```

### Install ikfast modules

To build ikfast modules:
```bash
$ cd utils/ikfast/kuka_kr6_r900/
$ python setup.py build
```

For other robots, replace `kuka_kr6_r900` with the following supported robots:
- `eth_rfl`
- `abb_irb6600_track`

### Installation test

1. Test pybullet: `python -c 'import pybullet'`

2. Testing new IKFast modules

* `$ python -m conrob_pybullet.debug_examples.test_eth_rfl_pick.py`
* `$ python -m conrob_pybullet.debug_examples.test_irb6600_track_pick.py`
