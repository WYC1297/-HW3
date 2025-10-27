@echo off
REM 快速部署到 GitHub 的 Windows 批處理腳本

echo 🚀 開始部署 SMS Spam Classifier 到 GitHub...

REM 檢查是否在正確的目錄
if not exist "streamlit_app.py" (
    echo ❌ 錯誤：請在項目根目錄執行此腳本
    pause
    exit /b 1
)

REM 初始化 Git（如果尚未初始化）
if not exist ".git" (
    echo 📦 初始化 Git 倉庫...
    git init
)

REM 添加所有文件
echo 📝 添加文件到 Git...
git add .

REM 提交變更
echo 💾 提交變更...
git commit -m "Deploy: SMS Spam Classifier with Enhanced Dashboard and Smart Example Generator"

REM 檢查是否已設置遠程倉庫
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo 🔗 請先設置 GitHub 遠程倉庫：
    echo    git remote add origin https://github.com/YOUR_USERNAME/sms-spam-classifier.git
    echo    然後重新運行此腳本
    pause
    exit /b 1
)

REM 推送到 GitHub
echo 🚀 推送到 GitHub...
git push -u origin main

echo ✅ 部署完成！
echo.
echo 📋 接下來的步驟：
echo 1. 前往 https://share.streamlit.io
echo 2. 使用 GitHub 帳號登錄
echo 3. 創建新應用，選擇您的倉庫
echo 4. 設置 Main file path: streamlit_app.py
echo 5. 自定義 App URL（建議：your-username-sms-spam-classifier）
echo.
echo 🌐 您的應用將在以下連結可用：
echo    https://your-username-sms-spam-classifier.streamlit.app/
echo.
pause