from setuptools import find_packages, setup
setup(name="qgloc", version="0.1.0",
      description="Localization radius estimation on the quasi-geostrophic model",
      packages=find_packages(include=["qgloc", "qgloc.*"]),
      python_requires=">=3.10",
      install_requires=["numpy>=1.26,<3", "scipy>=1.11", "pandas>=2.0",
                        "matplotlib>=3.7", "scikit-learn>=1.3"])
