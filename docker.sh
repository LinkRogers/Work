#!/bin/bash
docker run -p 8888:8888 -v $PWD:/tmp -w /tmp --rm -it linkrogers/research:tensorflow-jupyter-gpu bash
