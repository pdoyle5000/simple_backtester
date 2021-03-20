#!/bin/bash
black **/*.py
poetry run pytest --mypy
