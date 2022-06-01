from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="danbooru-utility",
    version="0.2.0",
    url="https://github.com/reidsanders/danbooru-utility.git",
    author="Reid Sanders",
    author_email="reid@reidsanders.net",
    description="Utility for working with danbooru2018 dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={"": ["danbooru-utility/lbpcascade_animeface.xml"]},
    packages=find_packages(),
    py_modules=["danbooru-utility/danbooru_utility"],
    install_requires=[
        "numpy >= 1.15.4",
        "opencv_python >= 3.4.3.18",
        "python_resize_image >= 1.1.18",
        "Pillow >= 5.4.1",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "danbooru-utility=danbooru.danbooru_utility:main",
        ],
    },
)
