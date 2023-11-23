#!/bin/bash

TMP_CODE_ARCHIVE_NAME=${1:-tmp-turkish-llm.tar.gz}

mkdir -p ~/turkish-llm && \
cd ~/turkish-llm && \
tar zxvf ~/${TMP_CODE_ARCHIVE_NAME} && \
cd -