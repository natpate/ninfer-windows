@echo off
rem ============================================================================
rem  download_model.bat
rem
rem  Downloads exactly ONE of the five published NInfer model artifacts from
rem  Hugging Face (huggingface.co) into the ./models directory that sits next
rem  to this script.
rem
rem  NInfer supports a deliberately closed set of model artifacts. The table
rem  below mirrors the README "Model artifacts" table (repo, file name, size,
rem  and the published SHA-256 used for the post-download integrity check).
rem
rem  Requirements: Windows 11 (curl.exe and certutil.exe ship in the box).
rem
rem  Usage:
rem      download_model.bat
rem  then pick a number. An interrupted download resumes on re-run; a complete
rem  file offers to be overwritten or is kept as-is.
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "MODELS=%ROOT%models"
if not exist "%MODELS%" mkdir "%MODELS%"

rem ---------------------------------------------------------------- require curl
where curl.exe >nul 2>nul
if errorlevel 1 (
    echo [error] curl.exe was not found on PATH.
    echo         Windows 11 ships curl.exe in the box; re-run from a full shell.
    echo.
    pause
    exit /b 1
)

:menu
cls
echo.
echo   NInfer - model artifact downloader  (Hugging Face)
echo   ====================================================
echo.
echo    [1]  Qwen3.6-27B        groupwise-int   qwen3_6_27b.ninfer         ~16.3 GiB
echo    [2]  Qwen3.6-27B NVFP4  nvfp4           qwen3_6_27b_nvfp4.ninfer   ~17.1 GiB
echo    [3]  Qwen3.8-27B        groupwise-int   qwen3_8_27b.ninfer          ~19.0 GiB
echo    [4]  Qwen3.8-27B NVFP4  nvfp4           qwen3_8_27b_nvfp4.ninfer   ~22.1 GiB
echo    [5]  Qwen3.6-35B-A3B    groupwise-int   qwen3_6_35b_a3b.ninfer     ~21.2 GiB
echo.
echo    [0]  cancel
echo.
set "CHOICE="
set /p "CHOICE=  pick one artifact [1-5]: "
if "%CHOICE%"=="" goto menu
if /i "%CHOICE%"=="0" goto :cancelled
if /i "%CHOICE%"=="q" goto :cancelled
if /i "%CHOICE%"=="quit" goto :cancelled
if /i "%CHOICE%"=="cancel" goto :cancelled

set "VALID="
for %%x in (1 2 3 4 5) do if "%CHOICE%"=="%%x" set "VALID=1"
if not defined VALID (
    echo.
    echo    !CHOICE!  - not a valid choice.
    echo.
    goto menu
)

rem ------------------------------------------------- resolve the chosen artifact
if "%CHOICE%"=="1" (
    set "LABEL=Qwen3.6-27B (groupwise-int)"
    set "REPO=neroued/Qwen3.6-27B-NInfer"
    set "FILE=qwen3_6_27b.ninfer"
    set "EXP_BYTES=17495365888"
    set "EXP_SHA=7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b"
) else if "%CHOICE%"=="2" (
    set "LABEL=Qwen3.6-27B (nvfp4)"
    set "REPO=neroued/Qwen3.6-27B-nvfp4-NInfer"
    set "FILE=qwen3_6_27b_nvfp4.ninfer"
    set "EXP_BYTES=18324064000"
    set "EXP_SHA=bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a"
) else if "%CHOICE%"=="3" (
    set "LABEL=Qwen3.8-27B (groupwise-int)"
    set "REPO=neroued/Qwen3.8-27B-NInfer"
    set "FILE=qwen3_8_27b.ninfer"
    set "EXP_BYTES=20437336576"
    set "EXP_SHA=0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3"
) else if "%CHOICE%"=="4" (
    set "LABEL=Qwen3.8-27B (nvfp4)"
    set "REPO=neroued/Qwen3.8-27B-nvfp4-NInfer"
    set "FILE=qwen3_8_27b_nvfp4.ninfer"
    set "EXP_BYTES=23719496192"
    set "EXP_SHA=552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c"
) else (
    set "LABEL=Qwen3.6-35B-A3B (groupwise-int)"
    set "REPO=neroued/Qwen3.6-35B-A3B-NInfer"
    set "FILE=qwen3_6_35b_a3b.ninfer"
    set "EXP_BYTES=22783246080"
    set "EXP_SHA=1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2"
)

set "OUT=%MODELS%\%FILE%"
set "URL=https://huggingface.co/%REPO%/resolve/main/%FILE%"

rem ---------------------------------------------------- existing-file handling
if not exist "%OUT%" goto :download
echo.
echo    Found existing file:  %OUT%
set "REDO="
set /p "REDO=  Re-download and overwrite? [y/N]: "
if /i "!REDO!"=="y" goto :redownload
goto :keep

:redownload
    del /f /q "%OUT%" 2>nul
:download
echo.
echo    Downloading    %LABEL%
echo    URL            %URL%
echo    Destination    %OUT%
echo    Expected size  %EXP_BYTES% bytes
echo.
curl.exe -L -C - --fail --retry 5 --retry-delay 3 -o "%OUT%" "%URL%"
if errorlevel 1 goto :dl_failed

rem ------------------------------------------------------- verify SHA-256
echo.
echo    Verifying SHA-256...
set "SHAFILE=%TMP%\hf_ninfer_%FILE%.sha"
certutil -hashfile "%OUT%" SHA256 > "%SHAFILE%" 2>nul
set "ACTUAL_SHA="
set /a _LN=0
for /f "usebackq delims=" %%l in ("%SHAFILE%") do (
    set /a _LN+=1
    if !_LN!==2 set "ACTUAL_SHA=%%l"
)
del "%SHAFILE%" 2>nul
if not defined ACTUAL_SHA (
    echo    [warn] Could not compute SHA-256 (certutil failed); skipping integrity check.
) else if /i "%ACTUAL_SHA%"=="%EXP_SHA%" (
    echo    [ok] SHA-256 verified:  %ACTUAL_SHA%
) else (
    echo    [warn] SHA-256 MISMATCH - the file may be corrupt, or the published
    echo           artifact changed since the README table was written.
    echo           expected:  %EXP_SHA%
    echo           actual:    %ACTUAL_SHA%
)
goto :success

:dl_failed
    echo.
    echo    [error] Download failed (curl exit %errorlevel%).
    echo    A partial file was kept for resume at:
    echo           %OUT%
    echo    Re-run this script and pick the same artifact to resume.
    goto :bye

:success
    echo.
    echo    Done. Your artifact is at:
    echo       %OUT%
    echo    (expected %EXP_BYTES% bytes)
    echo.
    echo    Start it with the matching launcher (qwen3_*.bat) or directly:
    echo       ninfer-serve.exe  models\%FILE%
    goto :bye

:keep
    echo.
    echo    Keeping the existing file. Nothing was downloaded.
    goto :bye

:cancelled
    echo.
    echo    Cancelled.
    goto :bye

:bye
    echo.
    pause
    exit /b 0
