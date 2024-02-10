#!/bin/bash

TMP_CODE_ARCHIVE_NAME=${1:-tmp-turna.tar.gz}

mkdir -p ~/turna && \
cd ~/turna && \
tar zxvf ~/${TMP_CODE_ARCHIVE_NAME} && \
cd -
