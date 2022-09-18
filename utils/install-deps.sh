#!/bin/bash

set -xe

# install azcopy
pushd /tmp
curl -sSL https://aka.ms/downloadazcopy-v10-linux -o azcopy.tar
tar -xvf azcopy.tar
cp azcopy_*/azcopy /usr/bin
chmod +x /usr/bin/azcopy
rm azcopy* -rf
popd