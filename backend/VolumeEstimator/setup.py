import logging
from setuptools import setup, find_packages

# Load requirements from file
try:
    with open('requirements.txt', 'r') as req_file:
        install_reqs = req_file.read()
except Exception:
    logging.warning('[!] Failed at loading requirements file.')

setup(
    name='food-volume-estimation',
    description='Estimate food volume from input image.',
    install_requires=install_reqs,
    python_requires='>=3.6',
    keywords='food volume estimation tensorflow keras',
    packages=find_packages()
)

