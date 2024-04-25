#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Need GITHUB User.email"
fi

echo "---"

git config credential.helper 'cache --timeout 7200'

git config --global user.email $1

source ./aliases.sh