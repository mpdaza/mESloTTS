import os 
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

cwd = os.path.dirname(os.path.abspath(__file__))

# Read requirements from environment.yml
def parse_requirements(filename):
    with open(filename, 'r') as file:
        requirements = []
        for line in file:
            if line.strip().startswith('- pip:'):
                break
        for line in file:
            if line.strip() and not line.strip().startswith('-'):
                requirements.append(line.strip())
    return requirements

reqs = parse_requirements('environment.yml')

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        os.system('python -m unidic download')

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        os.system('python -m unidic download')

setup(
    name='melotts',
    version='0.1.2',
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
    package_data={
        '': ['*.txt', 'cmudict_*'],
    },
    entry_points={
        "console_scripts": [
            "melotts = melo.main:main",
            "melo = melo.main:main",
            "melo-ui = melo.app:main",
        ],
    },
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)