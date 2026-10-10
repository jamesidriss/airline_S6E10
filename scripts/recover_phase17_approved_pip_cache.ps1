param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$Report,
    [Parameter(Mandatory=$true)][switch]$Approved
)
$ErrorActionPreference = 'Stop'
if (-not $Approved) { throw 'Explicit approval required for the exact shared-cache manifest.' }
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskCache = [IO.Path]::GetFullPath('C:\Users\James\AppData\Local\pip\Cache').TrimEnd('\')
$taskManifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$taskReportPath = [IO.Path]::GetFullPath((Join-Path $taskRoot $Report))
if (-not $taskManifestPath.StartsWith(($taskRoot + '\reports\'), [StringComparison]::OrdinalIgnoreCase)) { throw 'Manifest must be in this project reports directory.' }
if (-not $taskReportPath.StartsWith(($taskRoot + '\reports\'), [StringComparison]::OrdinalIgnoreCase)) { throw 'Report must be in this project reports directory.' }
if (Test-Path -LiteralPath $taskReportPath) { throw 'Preserve the previous recovery report.' }
$taskProposal = Get-Content -LiteralPath $taskManifestPath -Raw | ConvertFrom-Json
if ($taskProposal.operation_root -ne $taskCache -or $taskProposal.file_count -ne 2375) { throw 'The approved scope differs from the reviewed manifest.' }
$taskInstallers = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^pip(3)?\.exe$' -or ($_.Name -match '^python.*\.exe$' -and $_.CommandLine -match '\s-m\s+pip\s') })
if ($taskInstallers.Count -gt 0) { throw 'An installer is active; no cache operation is safe yet.' }
$taskVerified = @()
# Verify every absolute target and every original hash before deleting any file.
foreach ($taskFile in $taskProposal.files) {
    $taskPath = [IO.Path]::GetFullPath($taskFile.path)
    if (-not $taskPath.StartsWith(($taskCache + '\'), [StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside the approved cache directory.' }
    $taskResolved = (Resolve-Path -LiteralPath $taskPath).Path
    if ($taskResolved -ne $taskPath) { throw 'Resolved target changed.' }
    $taskItem = Get-Item -LiteralPath $taskPath -Force
    if ($taskItem.PSIsContainer -or ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Only approved regular files are eligible.' }
    if ($taskItem.Length -ne $taskFile.logical_bytes) { throw 'Approved file size changed.' }
    $taskHash = (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskHash -ne $taskFile.sha256) { throw 'Approved file content changed.' }
    $taskHandle = [IO.File]::Open($taskPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
    $taskHandle.Dispose()
    $taskVerified += [pscustomobject]@{path=$taskPath;sha256=$taskHash;logical_bytes=$taskItem.Length;exclusive_read_verified=$true}
}
$taskDrive = [IO.DriveInfo]::new('C:\')
$taskBefore = $taskDrive.AvailableFreeSpace
$taskStarted = [DateTime]::UtcNow
$taskDeleted = [Collections.Generic.List[object]]::new()
$taskState = [ordered]@{utc=$taskStarted.ToString('o');status='APPROVED_RECOVERY_IN_PROGRESS';approval='Human explicitly approved exact2375file manifest in this chat';manifest=$taskManifestPath;manifest_sha256=(Get-FileHash -LiteralPath $taskManifestPath -Algorithm SHA256).Hash.ToLowerInvariant();root=$taskCache;file_count_verified=$taskVerified.Count;disk_before_bytes=$taskBefore;deleted_files=@();scope='Exact regular pip cache files only; no recursive directory deletion or installed/model/personal/system file changes'}
$taskState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskReportPath -Encoding utf8
try {
    foreach ($taskFile in $taskVerified) {
        # Recheck that the file is not in use immediately before its literal deletion.
        $taskHandle = [IO.File]::Open($taskFile.path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
        $taskHandle.Dispose()
        Remove-Item -LiteralPath $taskFile.path -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $taskFile.path) { throw 'Deletion did not remove the approved file.' }
        $taskDeleted.Add($taskFile)
        if ($taskDeleted.Count % 100 -eq 0) {
            $taskState.deleted_files = $taskDeleted.ToArray()
            $taskState.disk_after_bytes = $taskDrive.AvailableFreeSpace
            $taskState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskReportPath -Encoding utf8
        }
    }
    $taskState.status = 'COMPLETE_APPROVED_EXACT_CACHE_RECOVERY'
} catch {
    $taskState.status = 'INCOMPLETE_RECOVERY_PRESERVE_MANIFEST'
    $taskState.error_type = $_.Exception.GetType().Name
    throw
} finally {
    $taskState.deleted_files = $taskDeleted.ToArray()
    $taskState.disk_after_bytes = $taskDrive.AvailableFreeSpace
    $taskState.measured_free_space_gain_bytes = $taskState.disk_after_bytes - $taskBefore
    $taskState.seconds = ([DateTime]::UtcNow - $taskStarted).TotalSeconds
    $taskState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskReportPath -Encoding utf8
    [pscustomobject]@{status=$taskState.status;deleted_files=$taskDeleted.Count;free_before_gib=$taskBefore/1GB;free_after_gib=$taskState.disk_after_bytes/1GB;measured_gain_gib=$taskState.measured_free_space_gain_bytes/1GB} | ConvertTo-Json
}
