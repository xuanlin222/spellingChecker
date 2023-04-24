from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    'transformers',
    'tqdm',
    'torch>=1.6.0',
    'numpy',
    'jsonlines',
    'sentencepiece',
    'pytorch_pretrained_bert'
]

setup(
    name="neuspell",
    version="1.0.0",
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">3.5",
    install_requires=requirements,
    extras_require={
        "spacy": ["spacy"],
        "elmo": ["allennlp==1.5.0"],
        "noising": ["unidecode"],
        "flask": ["flask_cors"]
    },
    keywords="transformer networks neuspell neural spelling correction embedding PyTorch NLP deep learning"
)
