import setuptools

setuptools.setup(
    name="emphatic-lstd-experiments",
    version="0.1.0",
    url="https://github.com/rldotai/emphatic-lstd-experiments",

    author="rldotai",
    author_email="rldot41@gmail.com",

    description="Experimenting with emphasis and least-squares temporal difference learning",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
)
