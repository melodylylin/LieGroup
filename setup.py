import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='liegrouppy',
    version='0.0.1',
    author='Melody Lin',
    author_email='',
    description='Lie group implementation in python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/zp-yang/LieGroups',
    # project_urls = {
    #     "Bug Tracker": "https://github.com/Muls/toolbox/issues"
    # },
    license='MIT',
    packages=['lienp', 'lieca'],
    install_requires=['requests'],
)