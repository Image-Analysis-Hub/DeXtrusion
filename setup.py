from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='dextrusion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.0.9',
    description='DeXtrusion: automatic detection of cell extrusion in epithelial tissu',
      author='Gaëlle Letort and Alexis Villars',
      url='https://github.com/Image-Analysis-Hub/dextrusion',
      package_dir={'':'src'},
      packages=find_packages('src'),
    install_requires=[
        "matplotlib",
        "numpy<2",
        "opencv-python",
        "tifffile>=2022.2.2",
        "roifile",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "tensorflow<=2.10", 
        "protobuf==3.19",
        "ipython"
    ],
)

