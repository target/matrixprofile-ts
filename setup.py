import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matrixprofile-ts",
    version="0.0.8",
    author="Andrew Van Benschoten",
    author_email="avbs89@gmail.com",
    description="An Open Source Python Time Series Library For Motif Discovery using Matrix Profile",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/target/matrixprofile-ts",
    packages = ['matrixprofile'],
    install_requires=['numpy>=1.11.3'],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
