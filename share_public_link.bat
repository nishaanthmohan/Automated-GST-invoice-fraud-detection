@echo off
echo ======================================================================
echo Launching High-Speed Cloudflare Tunnel for Teammates
echo Project: AI-Driven GST Invoice Anomaly Detection
echo ======================================================================
echo.
if not exist cloudflared.exe (
    echo [Notice] cloudflared.exe not found. Downloading portable Cloudflare binary...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe', 'cloudflared.exe')"
    echo Download completed!
)
echo Starting secure public tunnel to http://localhost:8501...
echo Look for the link ending in .trycloudflare.com below!
echo.
cloudflared.exe tunnel --url http://localhost:8501
pause
