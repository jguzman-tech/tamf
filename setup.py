from setuptools import setup, find_packages

setup(name="tamf",
      packages=find_packages(),
      install_requires=["pandas",
                        "geopandas",
                        "shapely",
                        "matplotlib",
                        "scipy",
                        "seaborn"],
      version="0.1")
