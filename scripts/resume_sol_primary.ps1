$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$python = Join-Path $taskRoot '.venv\Scripts\python.exe'
$routeArgs = @('--arms','route','--batch-size','1024','--windows-mqa',
    '--reuse-query-output','--decoder-inplace-gelu','--memory-saving','on',
    '--icl-bf16','--chunk-cells','262144','--col-chunk','1',
    '--host-reserve-gib','4','--prediction-host-reserve-gib','2',
    '--tag','sol_tabpfn35_route_recovery')
$jobs = @(
    @{Args=@('-u','scripts/run_sol_tabpfn.py','--folds','3')+$routeArgs; Log='reports/sol_route_recovery_f3.log'},
    @{Args=@('-u','scripts/run_sol_tabpfn.py','--folds','4')+$routeArgs; Log='reports/sol_route_recovery_f4.log'},
    @{Args=@('-u','scripts/cache_sol_aux.py','--folds','3'); Log='reports/sol_aux_cache_recovery_f3.log'},
    @{Args=@('-u','scripts/cache_sol_aux.py','--folds','4'); Log='reports/sol_aux_cache_recovery_f4.log'},
    @{Args=@('-u','scripts/replay_sol_isolated.py','--folds','0,1,2,3,4',
        '--tag','sol_clean_aux10_primary','--resume-from','sol_clean_aux10_threefold'); Log='reports/sol_clean_aux10_primary.log'},
    @{Args=@('scripts/evaluate_sol_foundation.py','--folds','0,1,2,3,4',
        '--classical-tag','sol_clean_aux10_primary',
        '--foundation-tags','sol_tabpfn35_route,sol_tabpfn35_route_reserve,sol_tabpfn35_route_recovery',
        '--name','sol_route_aux10_primary_final'); Log='reports/sol_route_aux10_primary_final.log'}
)
foreach ($job in $jobs) {
    if (Test-Path -LiteralPath $job.Log) { throw "Preserve existing log: $($job.Log); recover individual completed jobs explicitly" }
    $progress = @{Utc=(Get-Date).ToUniversalTime().ToString('o'); Status='RUNNING'; Log=$job.Log; Args=$job.Args}
    $progress | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'reports/sol_primary_recovery_progress.json' -Encoding utf8
    Write-Output ($progress.Utc + ' START ' + $job.Log)
    $arguments = $job.Args
    & $python @arguments *> $job.Log
    if ($LASTEXITCODE -ne 0) {
        $progress.Status='STOPPED_REVIEW_SAVED_REPORT'; $progress.ExitCode=$LASTEXITCODE
        $progress | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'reports/sol_primary_recovery_progress.json' -Encoding utf8
        exit $LASTEXITCODE
    }
    Write-Output ((Get-Date).ToUniversalTime().ToString('o') + ' COMPLETE ' + $job.Log)
}
@{Utc=(Get-Date).ToUniversalTime().ToString('o'); Status='COMPLETED_PRIMARY_REQUIRES_CONFIRMATION_AND_TEST'} |
    ConvertTo-Json | Set-Content -LiteralPath 'reports/sol_primary_recovery_progress.json' -Encoding utf8
