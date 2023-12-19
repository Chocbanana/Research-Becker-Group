#! /bin/bash

# If no  1st arg given, use default
DATA_PATH=${1:-'../../data/gen_plasma_n64/mat_2'}
# Get data folder name
NAME=$(basename $DATA_PATH)
ZIPNAME="${NAME}.zip"
# If no  2nd arg given, use default
TARGET_FOLDER=${2:-'../../tst'}

CURR_DIR="${PWD}"

# If zipped file doesn't exist, make it
if [ ! -f "${DATA_PATH}/../${ZIPNAME}" ]; then
    # Go to one directory below one we want to zip
    cd "${DATA_PATH}/../"
    echo "Zipping ${DATA_PATH}"
    # Zip folder recursively, update files if they exist, print progress dots
    # every 100MB
    zip -ru -qdgds 100m $ZIPNAME "$NAME"
fi

# Copy over zipped file to target folder, if it doesn't exist
if [ ! -f "${TARGET_FOLDER}/${ZIPNAME}" ]; then
    cp "${DATA_PATH}/../${ZIPNAME}" $TARGET_FOLDER
    echo "Unzipping ${NAME} into ${TARGET_FOLDER}"
    # Unzip files into target
    unzip -uq "${TARGET_FOLDER}/${ZIPNAME}" -d $TARGET_FOLDER
fi

cd $CURR_DIR

