from setuptools import setup, find_packages

setup(
    name='solar_tools',
    version='0.1.0',
    packages=find_packages(),  # This auto-discovers 'scripts'
    install_requires=[
        'matplotlib==3.9.0',
        'numpy==1.25.2',
        'pandas==2.2.3',
        'scikit-learn==1.6.1',
        'scipy==1.13.1',
        'seaborn==0.13.2',
        'windrose==1.9.2',
        'streamlit',
    ],
    author='Your Name',
    description='Solar energy data analysis tools',
    python_requires='>=3.9',
)
