from setuptools import setup
setup(name='Pynet',
	version='1.1.3',
	description='A neural network toolbox designed for evolution optimzation.',
	url='https://www.github.com/pmp47/Pynet',
	author='pmp47',
	author_email='phil@zeural.com',
	license='MIT',
	packages=['pynet'],
	install_requires=['numpy>=1.14.5','scikit-learn>=0.19.1','scipy>=1.1.0','tensorflow>=1.8.0'],
	zip_safe=False,
	include_package_data=True,
	python_requires='>=3.6',

	package_data={'': ['data/*.*']}
)
