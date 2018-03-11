"""Setup script for TensorFlow Agents"""

import setuptools

setuptools.setup(
  name="fake_agents",
  version="0.0.1",
  description=(
    "Fake reinforcement learning library",
  ),
  license="",
  url="https://github.com/ashigirl96/fake_agents",
  install_requires=[
    "tensorflow",
    "gym",
    "ruamel.yaml",
  ],
  packages=setuptools.find_packages(),
)