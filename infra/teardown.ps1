<#
  Tear down the demo to stop billing. By default deletes the whole resource group
  (fastest + cleanest). Use -KeepGroup to only delete the expensive pieces.
#>
param(
  [string]$ResourceGroup = "rg-tc-aml-foundry-etrm",
  [string]$Workspace     = "mlw-tc-etrm",
  [switch]$KeepGroup
)

$ErrorActionPreference = "Continue"

if (-not $KeepGroup) {
  Write-Host "Deleting resource group $ResourceGroup ..." -ForegroundColor Yellow
  az group delete -n $ResourceGroup --yes --no-wait
  Write-Host "Delete initiated (running in background)."
  return
}

Write-Host "Deleting online endpoint, compute, and web app (keeping group)..." -ForegroundColor Yellow
az ml online-endpoint delete -n etrm-forecast -w $Workspace -g $ResourceGroup --yes 2>$null
az ml compute delete -n cpu-cluster -w $Workspace -g $ResourceGroup --yes 2>$null
# Container Apps web app + its environment + registry
az containerapp delete -n etrm-qa -g $ResourceGroup --yes 2>$null
az containerapp env delete -n cae-tc-etrm -g $ResourceGroup --yes 2>$null
az acr delete -n acrtcetrm1c92281a -g $ResourceGroup --yes 2>$null
# Legacy App Service (if any remain)
az webapp list -g $ResourceGroup --query "[].name" -o tsv | ForEach-Object {
  az webapp delete -g $ResourceGroup -n $_ 2>$null
}
Write-Host "Done. Workspace + Foundry account retained."
