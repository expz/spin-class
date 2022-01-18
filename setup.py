from setuptools import setup


with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt", "r") as f:
    test_requirements = f.read().splitlines()

setup(
    name="spin-class",
    version="0.1.0",
    description="A library of simple RL algorithms written to spin up as an RL researcher.",
    author="Jonathan Skowera",
    author_email="jskowera@gmail.com",
    packages=["spin-class"],
    package_dir={"spin-class": "spin_class"},
    package_data={"spin-class": ["py.typed"]},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.7",
    test_suite="test",
    tests_requires=test_requirements,
)
