# Instructions for Maintainers

1. Update the version in `nlp4bia/__init__.py` and in `pyproject.toml`.
2. Remove the `dist` folder (`rm -rf dist`).
3. Build the package (`python -m build`).
4. Check the package (`twine check dist/*`).
5. Upload the package (`twine upload dist/*`).
6. Install the package (`pip install nlp4bia`).

Note: to build you have to install `build` and `twine` packages:
```bash
pip install build twine
```