from setuptools import setup, find_packages

setup(
    name='MSc-thesis-NESTAnets',
    version='1.0',
    description='Unrolled NESTA for Fourier imaging via TV minimization',
    url='https://github.com/mneyrane/MSc-thesis-NESTAnets',
    license='MIT',
    python_requires='>=3.9',
    install_requires=['torch', 'numpy', 'Pillow', 'scipy', 'matplotlib', 'seaborn'],
    packages=find_packages(),
)
