# Finite, frozen qualification queue. No upload or model changes in this controller.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($taskRoot -ne 'C:\Users\James\airline_S6E10') { throw 'Unexpected workspace' }
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'

function Invoke-FrozenStep {
    param([string]$Log, [string[]]$Arguments)
    if (Test-Path -LiteralPath $Log) { throw "Preserve existing log: $Log" }
    & $taskPython @Arguments > $Log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Frozen step failed; inspect $Log" }
}

$taskContractPath = 'reports\phase17_rescue_telemetry_B1_shadow3_retry_20261011\contract.json'
$taskContract = Get-Content -LiteralPath $taskContractPath -Raw | ConvertFrom-Json
$taskWorker = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskContract.worker_pid)"
if ($null -ne $taskWorker) {
    if ($taskWorker.CommandLine -notlike '*phase17_rescue_telemetry_B1_shadow3_retry_20261011*') { throw 'Worker identity mismatch' }
    Write-Output 'Waiting for the existing frozen B1 shadow3 worker; no model is launched concurrently.'
    Wait-Process -Id $taskWorker.ProcessId -ErrorAction SilentlyContinue
}
$taskSummary = Get-Content -LiteralPath 'reports\phase17_rescue_telemetry_B1_shadow3_retry_20261011\summary.json' -Raw | ConvertFrom-Json
if ($taskSummary.status -ne 'FROZEN_DRIVER_COMPLETE') { throw 'Shadow3 did not complete; no test inference' }
Invoke-FrozenStep 'reports\phase17_rescue_B1_shadow3_verification_20261011.log' @('-u','scripts\verify_phase17_rescue_context.py','--mode','shadow','--fold','3','--telemetry-tag','phase17_rescue_telemetry_B1_shadow3_retry_20261011','--output-tag','phase17_rescue_B1_shadow3_verification_20261011')
Write-Output 'MILESTONE: B1 shadow3 saved contributions replayed exactly; protected matched evaluation starts.'
Invoke-FrozenStep 'reports\phase17_rescue_shadow_gate_20261011.log' @('-u','scripts\evaluate_phase17_shadow.py','--primary-tag','phase17_B1','--candidate-tag','phase17_shadow_B1','--control-tag','phase17_shadow_B0_capacity','--auxiliary-tag','phase17_shadow_aux10')
$taskShadow = Get-Content -LiteralPath 'reports\phase17_B1_shadow_confirmation.json' -Raw | ConvertFrom-Json
if ($taskShadow.status -ne 'MATCHED_SHADOW_CONFIRMATION_PASS') { throw 'STOPPED: independent confirmation failed; no test inference or upload' }
Write-Output "MILESTONE: matched shadow PASS; frozen deltas $($taskShadow.paired_fold_deltas -join ', ')."
Invoke-FrozenStep 'reports\phase17_rescue_current_full_tests_20261011.log' @('-u','tests\run_tests.py')
Write-Output 'MILESTONE: current full tests passed; five exact test contexts run sequentially.'
foreach ($taskFold in 0..4) {
    $taskBudgetPath = if ($taskFold -eq 0) { 'reports\phase17_rescue_B1_shadow3_verification_20261011_compute.json' } else { "reports\phase17_rescue_test$($taskFold-1)_verification_20261011_compute.json" }
    $taskBudget = Get-Content -LiteralPath $taskBudgetPath -Raw | ConvertFrom-Json
    if ($taskBudget.remaining_model_hours -lt 1.5) { throw 'Remaining cumulative budget cannot cover the unchanged per-context runtime ceiling' }
    $taskTelemetry = "phase17_rescue_telemetry_test$($taskFold)_20261011"
    Invoke-FrozenStep "reports\phase17_rescue_test$($taskFold)_20261011.log" @('-u','scripts\run_phase17_monitored.py','--telemetry-tag',$taskTelemetry,'run_phase17_confirmation.py','--mode','test','--primary-tag','phase17_B1','--fold',"$taskFold",'--tag','phase17_B1_test')
    Invoke-FrozenStep "reports\phase17_rescue_test$($taskFold)_verification_20261011.log" @('-u','scripts\verify_phase17_rescue_context.py','--mode','test','--fold',"$taskFold",'--telemetry-tag',$taskTelemetry,'--output-tag',"phase17_rescue_test$($taskFold)_verification_20261011")
    Write-Output "MILESTONE: test context $taskFold complete; all299844 ordered rows and both saved members verified."
}
Invoke-FrozenStep 'reports\phase17_rescue_test_certificate_20261011.log' @('-u','scripts\assemble_phase17_test.py','--primary-tag','phase17_B1','--test-tag','phase17_B1_test','--name','v7_sol_tabpfn_internal2_aux10')
Write-Output 'MILESTONE: all five contexts certified and unique v7 CSV built; no upload yet.'
Invoke-FrozenStep 'reports\phase17_rescue_certificate_private820_20261011.log' @('-u','scripts\simulate_phase17_private.py','--certificate','reports\v7_sol_tabpfn_internal2_aux10_test_certificate.json','--tag','phase17_rescue_v7_private820_20261011')
Invoke-FrozenStep 'reports\phase17_rescue_v7_decision_20261011.log' @('-u','scripts\decide_phase17_submission.py','--certificate','reports\v7_sol_tabpfn_internal2_aux10_test_certificate.json','--simulation','reports\phase17_rescue_v7_private820_20261011.json','--role','A')
$taskDecision = Get-Content -LiteralPath 'reports\v7_sol_tabpfn_internal2_aux10_decision.json' -Raw | ConvertFrom-Json
Write-Output "MILESTONE: protected final decision $($taskDecision.verdict); root must audit, freeze and push pre-result evidence before the single authorized upload."
