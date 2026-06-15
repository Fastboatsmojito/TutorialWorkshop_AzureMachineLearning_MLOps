<#
.SYNOPSIS
  Build the Q&A web app image in ACR and deploy it to Azure Container Apps.

.DESCRIPTION
  Reproducible deployment of the FastAPI + Foundry-agent web app to Azure
  Container Apps. Uses an ACR cloud build (no local Docker required) and a
  user-assigned managed identity for BOTH registry pull and Foundry (AOAI)
  access via Entra ID (no API keys).

  Idempotent: re-running rebuilds the image with a new tag and rolls the app.

.NOTES
  Run AFTER provision.ps1 + the model online endpoint exist. Reads resource
  names / endpoint values from .azure-resources.json at the repo root.
#>
param(
  [string]$ResourceGroup = "rg-tc-aml-foundry-etrm",
  [string]$Location      = "eastus2",
  [string]$AcrName       = "acrtcetrm1c92281a",
  [string]$EnvName       = "cae-tc-etrm",
  [string]$AppName       = "etrm-qa",
  [string]$IdentityName  = "id-etrm-webapp"
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null   # avoid Windows cp1252 console crash while az streams logs

$repoRoot = Split-Path -Parent $PSScriptRoot
$res = Get-Content (Join-Path $repoRoot ".azure-resources.json") | ConvertFrom-Json
$loginServer = "$AcrName.azurecr.io"
$tag = "v$(Get-Date -Format 'yyyyMMddHHmm')"
$image = "$loginServer/etrm-webapp:$tag"

Write-Host "==> Ensuring ACR + managed identity exist"
az acr create -n $AcrName -g $ResourceGroup -l $Location --sku Basic -o none 2>$null
$uami = az identity create -n $IdentityName -g $ResourceGroup -l $Location -o json | ConvertFrom-Json

Write-Host "==> Assigning roles to identity (AcrPull + OpenAI User)"
$acrId = az acr show -n $AcrName -g $ResourceGroup --query id -o tsv
$foundryId = az cognitiveservices account show -n $res.foundry_account -g $ResourceGroup --query id -o tsv
az role assignment create --assignee-object-id $uami.principalId --assignee-principal-type ServicePrincipal --role "AcrPull" --scope $acrId -o none 2>$null
az role assignment create --assignee-object-id $uami.principalId --assignee-principal-type ServicePrincipal --role "Cognitive Services OpenAI User" --scope $foundryId -o none 2>$null

Write-Host "==> Bundling model metadata into the image build context"
Copy-Item (Join-Path $repoRoot "model_artifacts\model_card.json")          (Join-Path $repoRoot "webapp\backend\model_card.json") -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $repoRoot "model_artifacts\feature_importances.json") (Join-Path $repoRoot "webapp\backend\feature_importances.json") -Force -ErrorAction SilentlyContinue

Write-Host "==> Building image in ACR (cloud build): $image"
# The az CLI log stream can crash on Windows (cp1252); the cloud build still
# completes, so suppress the stream and poll the run result instead.
az acr build -r $AcrName -t "etrm-webapp:$tag" -f "$repoRoot\webapp\Dockerfile" "$repoRoot\webapp" 2>$null | Out-Null
do {
  Start-Sleep -Seconds 15
  $run = az acr task list-runs -r $AcrName --top 1 --query "[0].{status:status,tag:outputImages[0].tag}" -o json | ConvertFrom-Json
  Write-Host "    build status=$($run.status)"
} while ($run.status -in @("Running","Queued","Started"))
if ($run.status -ne "Succeeded") { throw "ACR build failed: $($run.status)" }

Write-Host "==> Creating Container Apps environment (if needed)"
az containerapp env create -n $EnvName -g $ResourceGroup -l $Location -o none 2>$null

$envVars = @(
  "AOAI_ENDPOINT=$($res.aoai_endpoint)",
  "AOAI_DEPLOYMENT=$($res.aoai_deployment)",
  "AOAI_API_VERSION=$($res.aoai_api_version)",
  "AML_ENDPOINT_URL=$($res.aml_endpoint_url)",
  "AML_ENDPOINT_KEY=$($res.aml_endpoint_key)",
  "AZURE_CLIENT_ID=$($uami.clientId)"
)

$exists = az containerapp show -n $AppName -g $ResourceGroup --query name -o tsv 2>$null
if ($exists) {
  Write-Host "==> Updating existing container app to $tag"
  az containerapp update -n $AppName -g $ResourceGroup --image $image --set-env-vars $envVars -o none
} else {
  Write-Host "==> Creating container app"
  az containerapp create -n $AppName -g $ResourceGroup `
    --environment $EnvName --image $image `
    --user-assigned $uami.id `
    --registry-server $loginServer --registry-identity $uami.id `
    --ingress external --target-port 8000 `
    --min-replicas 1 --max-replicas 2 --cpu 0.5 --memory 1.0Gi `
    --env-vars $envVars -o none
}

$fqdn = az containerapp show -n $AppName -g $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv
Write-Host "`n==> Deployed. Web app: https://$fqdn/"
