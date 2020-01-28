from setuptools import setup
setup(name='Pynet',
	version='1.4.8',
	description='A neural network toolbox designed for evolution optimzation.',
	url='https://www.github.com/pmp47/Pynet',
	author='pmp47',
	author_email='phil@zeural.com',
	license='MIT',
	packages=['pynet'],
	install_requires=['evolution @ git+https://github.com/pmp47/Evolution@master#egg=Evolution==1.1.1','numpyextension @ git+https://github.com/pmp47/NumpyExtension@master#egg=NumpyExtension==1.1.1','numpy==1.17.1','scikit-learn==0.21.3','scipy==1.3.1','tensorflow==1.15.2'],
	zip_safe=False,
	include_package_data=True,
	python_requires='>=3.6',

	package_data={'': ['data/*.*']}
)
