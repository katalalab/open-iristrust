param([string]$Destination = (Join-Path $PSScriptRoot '..\models\iris-semseg.onnx'))
$ErrorActionPreference = 'Stop'
$expected = '1228a1c760791749fceeedf9322e24f098aac80234368e1beee156e1e8fa57e7'
$url = 'https://huggingface.co/Worldcoin/iris-semantic-segmentation/resolve/e6c8fbb8e2e7e024a58818530a6e3ae5c005f47a/iris_semseg_upp_scse_mobilenetv2.onnx'
$destinationPath = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $destinationPath) {
    if ((Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw 'Existing model has an unexpected checksum; choose another destination.'
    }
    Write-Output "Verified model: $destinationPath"
    exit 0
}
New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destinationPath)) -Force | Out-Null
$temporaryPath = "$destinationPath.$([guid]::NewGuid().ToString('N')).part"
try {
    & curl.exe --fail --location --retry 3 --output $temporaryPath $url
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
    if ((Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw 'Downloaded model checksum mismatch'
    }
    Move-Item -LiteralPath $temporaryPath -Destination $destinationPath
    Write-Output "Verified model: $destinationPath"
} finally {
    if (Test-Path -LiteralPath $temporaryPath) { Remove-Item -LiteralPath $temporaryPath }
}
