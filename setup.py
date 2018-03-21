from setuptools import setup

setup(
    name='irf_builder',
    version='0.5',
    description='IRF builder for CTA',
    author='Tino Michael',
    author_email='tino.michael@cea.fr',
    packages=['irf_builder'],
    scripts=['scripts/make_point-source_irfs.py',
             'scripts/generate_toy_experiments.py'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'astropy', 'pandas',
                      'pyyaml', 'matplotlib', 'gammapy']
)
