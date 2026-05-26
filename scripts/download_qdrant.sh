#!/bin/bash
# Downloads the standalone Qdrant binary for Apple Silicon (M1/M2/M3/M4)
# Keeps the system clean by saving it only inside the project folder.

set -e

VERSION="v1.13.0"
OS_ARCH="aarch64-apple-darwin"
TAR_FILE="qdrant-${OS_ARCH}.tar.gz"
DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/${VERSION}/${TAR_FILE}"

echo "Downloading Qdrant ${VERSION} for Apple Silicon..."
mkdir -p bin
cd bin

if [ -f "qdrant" ]; then
    echo "✅ Qdrant binary already exists in bin/"
    exit 0
fi

curl -L -O ${DOWNLOAD_URL}
echo "Extracting..."
tar -xzf ${TAR_FILE}
rm ${TAR_FILE}

echo "✅ Qdrant successfully installed to arxiv-scholar/bin/qdrant"
echo "To start it, run: ./bin/qdrant"
