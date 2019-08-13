#!/bin/bash
docker run --runtime=nvidia -p 8888:8888 -v $PWD:/tmp -w /tmp --rm -it jupyter:gpu bash
