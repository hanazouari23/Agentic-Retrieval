# Regenerate 50-query traces for all policies on ms-marco-v2.1-segmented
$ErrorActionPreference = "Stop"
$workdir = "C:\Users\hanaz\Documents\GitHub\Agentic-Retrieval"
Set-Location $workdir

$outDir = "RQs/relevant_csvs"
$files = @(
    "cql_full_tree_traces_50_msmarco_dev.csv",
    "cql_full_tree_stop_penalty_traces_50_msmarco_dev.csv",
    "stop_traces_50_msmarco_dev.csv",
    "rerank_traces_50_msmarco_dev.csv",
    "reformulate_traces_50_msmarco_dev.csv",
    "expert_traces_50_msmarco_dev.csv"
)

Write-Host "Removing old trace files in $outDir ..."
foreach ($f in $files) {
    $p = Join-Path $outDir $f
    if (Test-Path $p) {
        Remove-Item -LiteralPath $p -Force
        Write-Host "  removed $p"
    }
}

$python = ".venv\Scripts\python.exe"

Write-Host "`n[1/3] Generating normal CQL full-tree traces ..."
& $python -m evaluation.generate_cql_traces_param `
    --model-path models/cql_on_full_tree_dataset.pt `
    --hidden-dims 256 128 `
    --reward-alpha 2.0 `
    --output-path RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv `
    --num-queries 50

Write-Host "`n[2/3] Generating CQL stop-penalty traces ..."
& $python -m evaluation.generate_cql_stop_penalty_traces

Write-Host "`n[3/3] Generating baseline traces ..."
& $python -m evaluation.generate_baseline_traces

Write-Host "`nRegeneration complete. Files in ${outDir}:"
Get-ChildItem -Path $outDir -Filter "*_traces_50_msmarco_dev.csv" | Select-Object Name, Length, LastWriteTime

Write-Host "`nRunning comparison ..."
& $python -m evaluation.compare_models
