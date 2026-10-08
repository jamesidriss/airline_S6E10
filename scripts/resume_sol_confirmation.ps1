$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$python=Join-Path $taskRoot '.venv\Scripts\python.exe'
$primary='reports/sol_route_aux10_primary_final.json'
$record=Get-Content -LiteralPath $primary -Raw | ConvertFrom-Json
if ($record.scope -ne 'full primary OOF' -or !$record.admission_gate_against_clean_auxiliary_stack.admit) {
    throw 'Full primary admission is required before confirmation'
}
$jobs=@(
    @{Args=@('-u','scripts/run_sol_tabpfn_confirm.py','--scheme','shadow','--fold','0','--tag','sol_tabpfn35_shadow_recovery'); Log='reports/sol_shadow_route_f0.log'},
    @{Args=@('-u','scripts/run_sol_tabpfn_confirm.py','--scheme','shadow','--fold','1','--tag','sol_tabpfn35_shadow_recovery'); Log='reports/sol_shadow_route_f1.log'},
    @{Args=@('-u','scripts/cache_sol_aux.py','--scheme','shadow','--folds','0'); Log='reports/sol_shadow_aux_cache_f0.log'},
    @{Args=@('-u','scripts/cache_sol_aux.py','--scheme','shadow','--folds','1'); Log='reports/sol_shadow_aux_cache_f1.log'},
    @{Args=@('-u','scripts/replay_sol_isolated.py','--confirmation-shadow','--folds','0,1','--tag','sol_clean_aux10_shadow_recovery'); Log='reports/sol_clean_aux10_shadow_recovery.log'},
    @{Args=@('scripts/evaluate_sol_foundation.py','--scheme','shadow','--folds','0,1',
        '--classical-tag','sol_clean_aux10_shadow_recovery','--foundation-tags','sol_tabpfn35_shadow_recovery',
        '--name','sol_route_aux10_shadow_recovery'); Log='reports/sol_shadow_portfolio_evaluation.log'},
    @{Args=@('-u','scripts/run_sol_tabpfn_confirm.py','--scheme','primary','--fold','0',
        '--seed-diagnostic','--tag','sol_tabpfn35_seed1202_recovery'); Log='reports/sol_seed1202_recovery.log'}
)
foreach ($job in $jobs) {
    if (Test-Path -LiteralPath $job.Log) { throw "Preserve existing log: $($job.Log)" }
    $progress=@{Utc=(Get-Date).ToUniversalTime().ToString('o'); Status='RUNNING'; Args=$job.Args; Log=$job.Log}
    $progress | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'reports/sol_confirmation_recovery_progress.json' -Encoding utf8
    Write-Output ($progress.Utc+' START '+$job.Log)
    $arguments=$job.Args
    & $python @arguments *> $job.Log
    if ($LASTEXITCODE -ne 0) { Write-Output ('STOP '+$job.Log); exit $LASTEXITCODE }
    Write-Output ((Get-Date).ToUniversalTime().ToString('o')+' COMPLETE '+$job.Log)
}
$shadow=Get-Content -LiteralPath 'reports/sol_route_aux10_shadow_recovery.json' -Raw | ConvertFrom-Json
if ($shadow.positive_folds -ne 2 -or $shadow.mean_paired_gain_vs_strict_aux10 -lt 0.000015) {
    throw 'Frozen partial shadow contradicts promotion; investigate before test inference'
}
@{Utc=(Get-Date).ToUniversalTime().ToString('o'); Status='COMPLETED_TWO_SHADOW_FOLDS_AND_FIXED_SEED_REQUIRES_TEST'} |
    ConvertTo-Json | Set-Content -LiteralPath 'reports/sol_confirmation_recovery_progress.json' -Encoding utf8
