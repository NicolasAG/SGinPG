#!/usr/bin/env bash

# Download all data files

if [[ ! -d "./backward" ]] || [[ ! -d "./forward" ]]
then
    echo "downloading clutrr data files..."
    export fileid=1TUXEcyR5i3TCrYtlTIX65swQH56TSiJj
    export filename=clutrr.zip
    curl -L -c cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
    curl -L -b cookies.txt -o $filename 'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
    rm confirm.txt cookies.txt
    echo "unzipping..."
    unzip -qq $filename
    rm $filename
    echo "done."
fi
