from setuptools import setup, find_packages

setup(
    name='FairModel',
    version='0.1.0',
    author='Johan van Soest',
    author_email='j.vansoest@maastrichtuniversity.nl',
    packages=find_packages(),
    url='https://github.com/MaastrichtU-BISS/FAIRmodels-model-package',
    license='Apache 2.0',
    description='A package to package AI models into an executionable container',
    long_description="A package to package AI models into an executionable container",
    install_requires=[
        "docker",
        "click"
    ],
    entry_points = {
        'console_scripts': [
            'fm build = LinkedDicom.cli:wrap',
            'fm predict = LinkedDicom.cli:predict'
        ]
    }
)