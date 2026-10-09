#!/bin/sh
set -eu
destination="${1:-models/iris-semseg.onnx}"
expected=1228a1c760791749fceeedf9322e24f098aac80234368e1beee156e1e8fa57e7
url=https://huggingface.co/Worldcoin/iris-semantic-segmentation/resolve/e6c8fbb8e2e7e024a58818530a6e3ae5c005f47a/iris_semseg_upp_scse_mobilenetv2.onnx
check() {
  if command -v sha256sum >/dev/null 2>&1; then printf '%s  %s\n' "$expected" "$1" | sha256sum --check --status
  else [ "$(shasum -a 256 "$1" | cut -d ' ' -f 1)" = "$expected" ]; fi
}
if [ -e "$destination" ]; then check "$destination"; printf 'Verified model: %s\n' "$destination"; exit 0; fi
mkdir -p "$(dirname "$destination")"
temporary=$(mktemp "${destination}.XXXXXX")
trap 'rm -f "$temporary"' EXIT HUP INT TERM
curl --fail --location --retry 3 --output "$temporary" "$url"
check "$temporary"
# Hard link refuses to replace a file concurrently created at the destination.
ln "$temporary" "$destination"
printf 'Verified model: %s\n' "$destination"
