from pathlib import Path

from setuptools import setup, find_packages


def check_directory():
    """
    You must always change directory to the parent of this file before
    executing the setup.py script. setuptools will fail reading files,
    including and excluding files from the MANIFEST.in, defining the library
    path, etc, if not.
    """
    from os import chdir

    here = Path(__file__).parent.resolve()
    if Path.cwd().resolve() != here:
        print('Changing path to {}'.format(here))
        chdir(str(here))


check_directory()


def read(filename):
    """
    Read a file relative to setup.py location.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, filename)) as fd:
        return fd.read()


def find_version(filename):
    """
    Find package version in file.
    """
    import re
    content = read(filename)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


def find_requirements(filename):
    """
    Find requirements in file.
    """
    import string
    content = read(filename)
    requirements = []
    for line in content.splitlines():
        line = line.strip()
        if line and line[:1] in string.ascii_letters:
            requirements.append(line)
    print(requirements)
    return requirements


setup(
    name='range_image_compression',
    version=find_version('src/range_image_compression/__init__.py'),
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={'': ['config/*']},
    include_package_data = True,

    # Dependencies
    install_requires=find_requirements('requirements.txt'),

    # Metadata
    author='Esteban Zamora Alvarado',
    author_email='esteban.zamora.al@gmail.com',
    description=(
        'range_image_compression is a python package that implements the deep learning'
        'based compression of the LiDAR range images.'
    ),
    long_description=read('README.md'),
    long_description_content_type='text/x-rst',

    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],

    # Entry points
    entry_points={
        'console_scripts': [
            'range_image_compression_train=range_image_compression.train:main',
        ],
    }
)
