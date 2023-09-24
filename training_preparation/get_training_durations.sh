#!/bin/bash

LOG_FILENAMES="$@"

AWK_FILES_DIR=$(dirname "$0")

for LOG_FILENAME in ${LOG_FILENAMES}; do
    begin_and_end_datetimes=$(awk -f ${AWK_FILES_DIR}/extract_begin_end_datetime.awk ${LOG_FILENAME})
    echo ${LOG_FILENAME} ${begin_and_end_datetimes}
done
