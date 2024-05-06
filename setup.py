from setuptools import setup

setup(
    name="module",
    version="0.1.0",
    description="An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models",
    packages=["module"],
    install_requires=[
        "torch",
        "transformers==4.16.2",
        "scipy",
        "scikit-learn",
        "nltk",
        "datasets==1.18.3",
        "accelerate==0.5.1",
    ],
    include_package_data=True,
    zip_safe=False,
)
