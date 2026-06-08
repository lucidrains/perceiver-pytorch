from setuptools import setup, find_packages

setup(
  name = 'perceiver-pytorch',
  packages = find_packages(),
  version = '0.9.0',
  license='MIT',
  description = 'Perceiver - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/perceiver-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformer',
    'attention mechanism'
  ],
  install_requires=[
    'einops>=0.8.0',
    'torch>=2.5'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
