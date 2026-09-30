# Run as Administrator. This changes only the WSL service startup mode and starts it.
# It does not restart Windows, install a distribution, or modify unrelated services.
$ErrorActionPreference = 'Stop'
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'Open PowerShell as Administrator, then run this script.' }
$service = Get-Service -Name WslService
if ($service.StartType -eq 'Disabled') { Set-Service -Name WslService -StartupType Manual }
Start-Service -Name WslService
wsl --status
wsl --list --verbose
