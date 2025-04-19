#!/bin/bash

python3 train.py
uvicorn src.app.api:app --host "0.0.0.0"
