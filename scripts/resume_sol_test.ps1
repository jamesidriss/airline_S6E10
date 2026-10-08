$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$python=Join-Path $taskRoot '.venv\Scripts\python.exe'
$primary='reports/sol_route_aux10_primary_final.json'
$record=Get-Content -LiteralPath $primary -Raw | ConvertFrom-Json
if ($record.scope -ne 'full primary OOF' -or !$record.admission_gate_against_clean_auxiliary_stack.admit) {
    throw 'Full primary admission required before test inference'
}
$shadow=Get-Content -LiteralPath 'reports/sol_route_aux10_shadow_recovery.json' -Raw | ConvertFrom-Json
if ($shadow.positive_folds -ne 2 -or $shadow.mean_paired_gain_vs_strict_aux10 -lt 0.000015) {
    throw 'Frozen shadow confirmation required before test inference'
}
$jobs=@()
foreach ($context in @('all','0','1','2','3','4')) {
    $jobs+=@{Args=@('-u','scripts/run_sol_policy_pseudotest.py','--context',$context); Log="reports/sol_policy_context_$context.log"}
}
$jobs+=@{Args=@('scripts/evaluate_sol_policy_pseudotest.py'); Log='reports/sol_policy_comparison.log'}
foreach ($fold in 0..4) {
    $jobs+=@{Args=@('-u','scripts/run_sol_tabpfn_fold_test.py','--primary-report',$primary,
        '--fold',"$fold",'--tag','sol_tabpfn35_cv_test_recovery'); Log="reports/sol_route_context_test_f$fold.log"}
}
$jobs+=@{Args=@('-u','scripts/cache_sol_aux.py','--test'); Log='reports/sol_full_test_aux_cache.log'}
$plan=Get-Content -LiteralPath 'reports/sol_clean_aux10_primary/frozen_roles.json' -Raw | ConvertFrom-Json
foreach ($role in $plan.roles | Where-Object { $_.aux_arm }) {
    $jobs+=@{Args=@('-u','scripts/run_sol_aux_test.py','--primary-report',$primary,'--member',$role.member);
        Log=('reports/sol_full_test_'+$role.member+'.log')}
}
$jobs+=@{Args=@('scripts/assemble_sol_foundation_test.py','--primary-report',$primary,
    '--foundation-tag','sol_tabpfn35_cv_test_recovery','--foundation-policy','context_average',
    '--name','sol_route_aux10_primary_final'); Log='reports/sol_frozen_test_assembly.log'}
foreach ($job in $jobs) {
    if (Test-Path -LiteralPath $job.Log) { throw "Preserve existing log: $($job.Log)" }
    $progress=@{Utc=(Get-Date).ToUniversalTime().ToString('o'); Status='RUNNING'; Args=$job.Args; Log=$job.Log}
    $progress | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'reports/sol_test_recovery_progress.json' -Encoding utf8
    Write-Output ($progress.Utc+' START '+$job.Log)
    $arguments=$job.Args
    & $python @arguments *> $job.Log
    if ($LASTEXITCODE -ne 0) { Write-Output ('STOP '+$job.Log); exit $LASTEXITCODE }
    Write-Output ((Get-Date).ToUniversalTime().ToString('o')+' COMPLETE '+$job.Log)
}
@{Utc=(Get-Date).ToUniversalTime().ToString('o'); Status='CERTIFIED_TEST_REQUIRES_PRIVATE_DIAGNOSTICS_AND_DECISION'} |
    ConvertTo-Json | Set-Content -LiteralPath 'reports/sol_test_recovery_progress.json' -Encoding utf8
