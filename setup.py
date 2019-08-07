#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import print_function

import io
from os import path

from setuptools import setup


here = path.abspath(path.dirname(__file__))


def read(*names, **kwargs):
    return io.open(
        path.join(here, *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


long_description = read('README.md')
requirements = read('requirements.txt').split('\n')
optional_requirements = {
    # "conmech" : ['conmech']
}

setup(
    name='pychoreo',
    version='0.0.1',
    description='Choreo: a sequence and motion planning algorithm for discrete architectural assembly',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yijiangh/pychoreo',
    author='Yijiang Huang',
    author_email='yijiangh@mit.edu',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['digital fabrication', 'robotics', 'assembly planning'],
    project_urls={
        "Repository": "https://github.com/yijiangh/pychoreo",
        "Issues": "https://github.com/yijiangh/pychoreo/issues",
    },
    packages=['choreo', 'conrob_pybullet'],
    # package_dir={'': ''},
    package_data={},
    data_files=[],
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    python_requires='>=2.7',
    # extras_require=optional_requirements,
    entry_points={},
    ext_modules=[],
    cmdclass={},
    scripts=[]
)
