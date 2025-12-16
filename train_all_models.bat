@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==========================================
echo MMDetection Batch Training Script
echo Dataset: coco8
echo Models: Faster R-CNN, Cascade R-CNN, Mask R-CNN, HTC, SCNet, Grid R-CNN
echo ==========================================
echo.

cd /d %~dp0

REM Create work_dirs
if not exist work_dirs mkdir work_dirs

REM Record start time
set START_TIME=%time%

set SUCCESS_COUNT=0
set FAIL_COUNT=0

echo.
echo [1/6] Training Faster R-CNN...
echo ----------------------------------------------
python tools/train.py coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/faster_rcnn_coco8
if %errorlevel% equ 0 (
    echo [SUCCESS] Faster R-CNN completed!
    set /a SUCCESS_COUNT+=1
) else (
    echo [FAILED] Faster R-CNN failed!
    set /a FAIL_COUNT+=1
)

echo.
echo [2/6] Training Cascade R-CNN...
echo ----------------------------------------------
python tools/train.py coco8_configs/configs/cascade_rcnn/cascade-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/cascade_rcnn_coco8
if %errorlevel% equ 0 (
    echo [SUCCESS] Cascade R-CNN completed!
    set /a SUCCESS_COUNT+=1
) else (
    echo [FAILED] Cascade R-CNN failed!
    set /a FAIL_COUNT+=1
)

echo.
echo [3/6] Training Mask R-CNN...
echo ----------------------------------------------
python tools/train.py coco8_configs/configs/mask_rcnn/mask-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/mask_rcnn_coco8
if %errorlevel% equ 0 (
    echo [SUCCESS] Mask R-CNN completed!
    set /a SUCCESS_COUNT+=1
) else (
    echo [FAILED] Mask R-CNN failed!
    set /a FAIL_COUNT+=1
)

echo.
echo [4/6] Training HTC...
echo ----------------------------------------------
python tools/train.py coco8_configs/configs/htc/htc_r50_fpn_2e_coco8.py --work-dir work_dirs/htc_coco8
if %errorlevel% equ 0 (
    echo [SUCCESS] HTC completed!
    set /a SUCCESS_COUNT+=1
) else (
    echo [FAILED] HTC failed!
    set /a FAIL_COUNT+=1
)

echo.
echo [5/6] Training SCNet...
echo ----------------------------------------------
python tools/train.py coco8_configs/configs/scnet/scnet_r50_fpn_2e_coco8.py --work-dir work_dirs/scnet_coco8
if %errorlevel% equ 0 (
    echo [SUCCESS] SCNet completed!
    set /a SUCCESS_COUNT+=1
) else (
    echo [FAILED] SCNet failed!
    set /a FAIL_COUNT+=1
)

echo.
echo [6/6] Training Grid R-CNN...
echo ----------------------------------------------
python tools/train.py coco8_configs/configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2e_coco8.py --work-dir work_dirs/grid_rcnn_coco8
if %errorlevel% equ 0 (
    echo [SUCCESS] Grid R-CNN completed!
    set /a SUCCESS_COUNT+=1
) else (
    echo [FAILED] Grid R-CNN failed!
    set /a FAIL_COUNT+=1
)

REM Record end time
set END_TIME=%time%

echo.
echo ==========================================
echo Training Summary
echo ==========================================
echo Start Time: %START_TIME%
echo End Time: %END_TIME%
echo Successful: %SUCCESS_COUNT% / 6
echo Failed: %FAIL_COUNT% / 6
echo.
echo Trained models saved in work_dirs/
echo ==========================================
dir work_dirs /b 2>nul
echo.
pause
