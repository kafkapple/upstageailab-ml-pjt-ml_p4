from setuptools import setup, find_packages

# requirements.txt 읽어서 install_requires에 포함
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="CrawlerService",
    version="1.0.0",
    author="dongwank",
    description="A Python package for Naver Blog crawling",
    packages=find_packages(),
    install_requires=required,  # requirements.txt 기반
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
