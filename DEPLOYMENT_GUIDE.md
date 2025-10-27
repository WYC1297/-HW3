# 🚀 部署到 GitHub 和 Streamlit Cloud 指南

## 📋 部署步驟

### 1. 準備 GitHub 倉庫

1. **創建 GitHub 倉庫**：
   - 登錄 GitHub
   - 點擊 "New repository"
   - 倉庫名稱建議：`sms-spam-classifier`
   - 設置為 Public（Streamlit Cloud 免費版需要公開倉庫）

2. **上傳代碼到 GitHub**：
```bash
# 初始化 Git 倉庫
cd c:\Users\WANG\Desktop\hw3
git init

# 添加遠程倉庫（替換為您的 GitHub 用戶名）
git remote add origin https://github.com/YOUR_USERNAME/sms-spam-classifier.git

# 添加所有文件
git add .

# 提交代碼
git commit -m "Initial commit: SMS Spam Classifier with Streamlit Dashboard"

# 推送到 GitHub
git push -u origin main
```

### 2. 部署到 Streamlit Cloud

1. **訪問 Streamlit Cloud**：
   - 前往 [share.streamlit.io](https://share.streamlit.io)
   - 使用 GitHub 帳號登錄

2. **創建新應用**：
   - 點擊 "New app"
   - 選擇您的 GitHub 倉庫：`YOUR_USERNAME/sms-spam-classifier`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: 自定義為 `your-username-sms-spam-classifier`

3. **配置部署**：
   - Streamlit Cloud 會自動安裝 `requirements.txt` 中的依賴
   - 首次部署可能需要 5-10 分鐘

### 3. 獲取您的專屬連結

部署完成後，您的應用將可在以下連結訪問：
```
https://your-username-sms-spam-classifier.streamlit.app/
```

## 🔧 部署配置文件說明

### `.streamlit/config.toml`
- Streamlit 應用的主題和配置
- 設置了專業的顏色主題

### `requirements.txt`
- 包含所有必要的 Python 依賴
- 已優化為雲端部署版本

### `.gitignore`
- 排除不必要的文件（虛擬環境、緩存等）
- 保持倉庫清潔

### `README.md`
- 項目的完整說明文檔
- 包含在線演示連結

## 📊 部署檢查清單

- [ ] GitHub 倉庫已創建並設為 Public
- [ ] 所有代碼已推送到 main 分支
- [ ] `requirements.txt` 包含所有依賴
- [ ] `streamlit_app.py` 在根目錄
- [ ] 模型文件已包含（如果大小允許）
- [ ] Streamlit Cloud 應用已創建
- [ ] 應用成功部署並可訪問

## 🎯 自定義您的連結

在 Streamlit Cloud 部署時，您可以自定義 URL：
- 建議格式：`your-username-sms-spam-classifier`
- 最終連結：`https://your-username-sms-spam-classifier.streamlit.app/`

## 📝 部署後更新

每當您推送新代碼到 GitHub main 分支時，Streamlit Cloud 會自動重新部署您的應用。

## 🚨 常見問題

### 模型文件太大
如果模型文件超過 GitHub 限制（100MB），可以：
1. 使用 Git LFS
2. 在應用中動態下載模型
3. 使用 Hugging Face Hub 存儲模型

### 依賴安裝失敗
確保 `requirements.txt` 中的版本號正確且兼容。

### NLTK 數據下載
應用會在首次運行時自動下載必要的 NLTK 數據。

## 🎉 完成！

部署完成後，您將擁有：
- 🌐 專業的在線 SMS 垃圾郵件分類器
- 📊 交互式數據可視化
- 🔮 實時預測功能
- 🎲 智能範例生成器
- 📈 全面的模型分析

您的連結：`https://your-username-sms-spam-classifier.streamlit.app/`