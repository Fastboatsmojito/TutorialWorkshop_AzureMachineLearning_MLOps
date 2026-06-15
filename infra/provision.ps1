<#
  Provision all Azure resources for the TC Energy ETRM demo.
  Idempotent: safe to re-run. Requires: az CLI logged in (az login) + az ml extension.

  Resources (all in rg-tc-aml-foundry-etrm / eastus2):
    - Azure ML workspace (mlw-tc-etrm) + storage/keyvault/appinsights
    - Azure AI Foundry (AIServices) account (foundry-tc-etrm) with:
        * gpt-4o deployment       (the app / agent model)
        * gpt-4o-eval deployment  (the evaluation judge model; isolates demo traffic)
        * a Foundry project        (so evaluation runs show in the Foundry portal)
        * Content Safety           (Prompt Shields + text moderation for guardrails)
    - AML compute cluster (cpu-cluster)

  The Q&A web app is deployed separately to Azure Container Apps via
  infra/deploy_containerapp.ps1 (ACR cloud build + managed identity).
#>
param(
  [string]$ResourceGroup = "rg-tc-aml-foundry-etrm",
  [string]$Location      = "eastus2",
  [string]$Workspace     = "mlw-tc-etrm",
  [string]$Foundry       = "foundry-tc-etrm",
  [string]$ModelName     = "gpt-4o",
  [string]$EvalModelName = "gpt-4o-eval",
  [string]$Project       = "proj-tc-etrm",
  [string]$Cluster       = "cpu-cluster"
)

$ErrorActionPreference = "Stop"
az configure --defaults group=$ResourceGroup location=$Location | Out-Null

Write-Host "==> Resource group" -ForegroundColor Cyan
az group create -n $ResourceGroup -l $Location | Out-Null

Write-Host "==> Azure ML workspace" -ForegroundColor Cyan
az ml workspace create -n $Workspace -g $ResourceGroup -l $Location 2>$null

Write-Host "==> Identity-based datastore auth (tenant disables storage keys)" -ForegroundColor Cyan
$me = az ad signed-in-user show --query id -o tsv
$saId = az ml workspace show -n $Workspace -g $ResourceGroup --query storage_account -o tsv
az role assignment create --assignee $me --role "Storage Blob Data Contributor" --scope $saId 2>$null
az ml workspace update -n $Workspace -g $ResourceGroup --system-datastores-auth-mode identity | Out-Null

Write-Host "==> Azure AI Foundry account" -ForegroundColor Cyan
az cognitiveservices account create -n $Foundry -g $ResourceGroup -l $Location `
  --kind AIServices --sku S0 --custom-domain $Foundry --yes 2>$null

Write-Host "==> gpt-4o deployment" -ForegroundColor Cyan
az cognitiveservices account deployment create -g $ResourceGroup -n $Foundry `
  --deployment-name $ModelName --model-name gpt-4o --model-version "2024-11-20" `
  --model-format OpenAI --sku-name GlobalStandard --sku-capacity 30 2>$null

Write-Host "==> gpt-4o-eval deployment (judge model for Foundry evaluations)" -ForegroundColor Cyan
# A SEPARATE deployment for evaluations so eval traffic never competes with the
# live demo's gpt-4o capacity. The tutorial's quality evaluators use this judge.
az cognitiveservices account deployment create -g $ResourceGroup -n $Foundry `
  --deployment-name $EvalModelName --model-name gpt-4o --model-version "2024-11-20" `
  --model-format OpenAI --sku-name GlobalStandard --sku-capacity 30 2>$null

# Grant yourself the data-plane role so AAD calls to the model work (keys are disabled).
$foundryId = az cognitiveservices account show -n $Foundry -g $ResourceGroup --query id -o tsv
az role assignment create --assignee $me --role "Cognitive Services OpenAI User" --scope $foundryId 2>$null
# Content Safety (Prompt Shields + moderation) and the RAI eval service authorize
# via the broader "Cognitive Services User" data-plane role.
az role assignment create --assignee $me --role "Cognitive Services User" --scope $foundryId 2>$null

Write-Host "==> Foundry project (evaluation runs appear here in the Foundry portal)" -ForegroundColor Cyan
# A project is a child resource of the AIServices account. Evaluations logged to
# the project endpoint show up under Foundry portal > your project > Evaluations.
az resource create -g $ResourceGroup `
  --namespace Microsoft.CognitiveServices --resource-type accounts/projects `
  --parent "accounts/$Foundry" -n $Project --location $Location `
  --properties '{\"displayName\":\"TC ETRM Forecast\",\"description\":\"ETRM forecasting agent evaluations and guardrails.\"}' `
  --api-version 2025-04-01-preview 2>$null
# Give yourself project data-plane access (run evals, write datasets/results).
$projectId = az resource show -g $ResourceGroup --namespace Microsoft.CognitiveServices `
  --resource-type accounts/projects --parent "accounts/$Foundry" -n $Project `
  --api-version 2025-04-01-preview --query id -o tsv 2>$null
az role assignment create --assignee $me --role "Azure AI Developer" --scope $foundryId 2>$null
$projectEndpoint = "https://$Foundry.services.ai.azure.com/api/projects/$Project"
$contentSafetyEndpoint = "https://$Foundry.cognitiveservices.azure.com/"

Write-Host "==> AML compute cluster (with managed identity for AutoML data access)" -ForegroundColor Cyan
az ml compute create -n $Cluster --type AmlCompute --min-instances 0 --max-instances 2 `
  --size Standard_DS3_v2 --identity-type SystemAssigned -w $Workspace -g $ResourceGroup 2>$null
# AutoML runs trials as the cluster's identity; it needs to read the dataset from blob.
$clusterPrincipal = az ml compute show -n $Cluster -w $Workspace -g $ResourceGroup --query "identity.principal_id" -o tsv 2>$null
if ($clusterPrincipal) {
  az role assignment create --assignee-object-id $clusterPrincipal --assignee-principal-type ServicePrincipal `
    --role "Storage Blob Data Contributor" --scope $saId 2>$null
  # Evaluations run on AML compute: the cluster identity calls the judge model,
  # the RAI safety service, and Content Safety, so grant it the data-plane roles.
  az role assignment create --assignee-object-id $clusterPrincipal --assignee-principal-type ServicePrincipal `
    --role "Cognitive Services OpenAI User" --scope $foundryId 2>$null
  az role assignment create --assignee-object-id $clusterPrincipal --assignee-principal-type ServicePrincipal `
    --role "Cognitive Services User" --scope $foundryId 2>$null
  az role assignment create --assignee-object-id $clusterPrincipal --assignee-principal-type ServicePrincipal `
    --role "Azure AI Developer" --scope $foundryId 2>$null
}

Write-Host "==> Writing Foundry/eval settings to .azure-resources.json" -ForegroundColor Cyan
# Merge (don't clobber) so endpoint URL/key written by deploy_endpoint.py survive.
$resourcesFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".azure-resources.json"
$res = @{}
if (Test-Path $resourcesFile) {
  (Get-Content $resourcesFile -Raw | ConvertFrom-Json).psobject.properties | ForEach-Object { $res[$_.Name] = $_.Value }
}
$res["subscription_id"]         = (az account show --query id -o tsv)
$res["resource_group"]          = $ResourceGroup
$res["workspace"]               = $Workspace
$res["location"]                = $Location
$res["compute_cluster"]         = $Cluster
$res["data_asset_name"]         = "aeso-hourly-prices"
$res["model_name"]              = "aeso-price-forecaster"
$res["aoai_endpoint"]           = "https://$Foundry.openai.azure.com/"
$res["aoai_deployment"]         = $ModelName
$res["aoai_eval_deployment"]    = $EvalModelName
$res["aoai_api_version"]        = "2024-10-21"
$res["foundry_account"]         = $Foundry
$res["foundry_project_name"]    = $Project
$res["foundry_project_endpoint"] = $projectEndpoint
$res["content_safety_endpoint"] = $contentSafetyEndpoint
$res | ConvertTo-Json -Depth 5 | Set-Content $resourcesFile

Write-Host "`nProvisioning complete." -ForegroundColor Green
Write-Host "Foundry endpoint:  https://$Foundry.openai.azure.com/"
Write-Host "Project endpoint:  $projectEndpoint"
Write-Host "Eval judge model:  $EvalModelName"
Write-Host "Next: run training + deploy_endpoint.py, then infra/deploy_containerapp.ps1 for the Q&A web app."
