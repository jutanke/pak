#from distutils.core import setup

#setup_req = ['numpy']
#install_req = ['numpy']

#setup(
#    name = 'pak',
#    packages = ['pak'],
#    install_requires=install_req,
#    setup_requires=setup_req,
#    version = '0.0.1',
#    description = 'General Dataset Helper Functions',
#    author = 'Julian Tanke',
#    url='https://github.com/justayak/pppr'
#)

from setuptools import setup, find_packages

print("PACKAGES:",find_packages())

setup(
    name="pak",
    version="0.0.2",
    packages=find_packages(),
)
