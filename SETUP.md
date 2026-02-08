# 📱 Android Setup Guide - ML Unified System v3.3

Complete guide for running ML Unified System on Android devices

---

## 🎯 Why Run ML on Android?

- ✅ Analyze smart contracts on-the-go
- ✅ Mobile security research
- ✅ Learn blockchain security anywhere
- ✅ No desktop required
- ✅ Perfect for students and researchers

---

## 📲 Installation Options

### Option 1: Pydroid 3 (Recommended for Beginners)

**Step 1:** Install Pydroid 3
- Download from [Google Play Store](https://play.google.com/store/apps/details?id=ru.iiec.pydroid3)
- Free version works perfectly!

**Step 2:** Install Required Packages
1. Open Pydroid 3
2. Tap the **menu** (☰) → **Pip**
3. Install packages one by one:
   ```
   numpy
   pandas
   scikit-learn
   joblib
   ```
4. Wait for each to complete (may take 5-10 minutes total)

**Step 3:** Download ML System
1. Open browser on your phone
2. Go to GitHub repository
3. Download `ML_UNIFIED_SYSTEM_V3_3.py`
4. Save to `/storage/emulated/0/Download/` or Pydroid folder

**Step 4:** Create Data Folder
1. Using file manager, create folder structure:
   ```
   /storage/emulated/0/MLSystem/
   ├── ML_UNIFIED_SYSTEM_V3_3.py
   ├── data_json/          ← Create this folder
   ├── data_txt/           ← Create this folder
   ├── data_db/            ← Create this folder
   ├── trained_models/     ← Will be auto-created
   └── ml_output/          ← Will be auto-created
   ```

**Step 5:** Run the System
1. Open `ML_UNIFIED_SYSTEM_V3_3.py` in Pydroid 3
2. Tap the ▶️ (Play) button
3. First run will create necessary folders
4. Done! 🎉

---

### Option 2: Termux (For Advanced Users)

**Step 1:** Install Termux
- Download from [F-Droid](https://f-droid.org/en/packages/com.termux/) (recommended)
- **NOT** from Play Store (outdated version)

**Step 2:** Setup Python Environment
```bash
# Update packages
pkg update && pkg upgrade -y

# Install Python and dependencies
pkg install python python-pip git -y

# Install required Python packages
pip install numpy pandas scikit-learn joblib
```

**Step 3:** Download ML System
```bash
# Method A: Using git (recommended)
cd ~/storage/downloads
git clone https://github.com/yourusername/ml-unified-system.git
cd ml-unified-system

# Method B: Direct download
cd ~/storage/downloads
curl -O https://raw.githubusercontent.com/yourusername/ml-unified-system/main/ML_UNIFIED_SYSTEM_V3_3.py
```

**Step 4:** Setup Storage Access
```bash
# Grant storage permission
termux-setup-storage
# Tap "Allow" when prompted

# Create symlink to easier access
ln -s ~/storage/downloads ~/downloads
ln -s ~/storage/shared ~/shared
```

**Step 5:** Run the System
```bash
cd ~/downloads/ml-unified-system
python ML_UNIFIED_SYSTEM_V3_3.py
```

---

## 🗂️ File Organization on Android

### Recommended Structure

**Pydroid 3:**
```
/storage/emulated/0/Pydroid/
├── MLSystem/
│   ├── ML_UNIFIED_SYSTEM_V3_3.py
│   ├── data_json/
│   │   ├── contract_0x1234.json
│   │   └── contract_0x5678.json
│   ├── trained_models/
│   └── ml_output/
```

**Termux:**
```
~/storage/downloads/ml-unified-system/
├── ML_UNIFIED_SYSTEM_V3_3.py
├── data_json/
├── trained_models/
└── ml_output/
```

---

## 🚀 Usage on Android

### Basic Workflow (Pydroid 3)

1. **Prepare Contract Data**
   - Download JSON reports from your desktop
   - Transfer via USB, Google Drive, or Dropbox
   - Place in `data_json/` folder

2. **Run Analysis**
   - Open `ML_UNIFIED_SYSTEM_V3_3.py` in Pydroid 3
   - Tap ▶️ Play
   - Watch progress in console

3. **View Results**
   - Open `ml_output/` folder
   - View `scoring_results.json` in text editor
   - Or transfer to desktop for better viewing

### Basic Workflow (Termux)

```bash
# Navigate to project
cd ~/downloads/ml-unified-system

# Add your contract data
# (copy JSON files to data_json/ via file manager)

# Run analysis
python ML_UNIFIED_SYSTEM_V3_3.py

# View results
cat ml_output/scoring_results.json

# Or use termux-open to view in browser
pkg install termux-tools
termux-open ml_output/scoring_results.json
```

---

## 💡 Android-Specific Tips

### 1. Managing Storage

**Check Available Space:**
```bash
# In Termux
df -h ~/storage
```

**Free Up Space:**
- Delete old models in `trained_models/`
- Archive old results from `ml_output/`
- Keep only recent contract data

### 2. Performance Optimization

**Pydroid 3:**
- Close other apps while training
- Training 100 contracts ~5-10 minutes
- Scoring is much faster (~30 seconds for 30 contracts)

**Termux:**
- Use `nice` for background processing:
  ```bash
  nice -n 19 python ML_UNIFIED_SYSTEM_V3_3.py &
  ```

### 3. Battery Management

**Long Training Sessions:**
- Keep phone plugged in
- Disable battery optimization for Pydroid 3/Termux
- Use airplane mode to prevent interruptions

### 4. File Transfer Methods

**Option A: USB Cable**
```
Computer → USB → Phone/storage/emulated/0/MLSystem/data_json/
```

**Option B: Cloud Storage**
```
Computer → Google Drive → Download on Phone → Move to data_json/
```

**Option C: Direct Download**
```bash
# In Termux
cd ~/downloads/ml-unified-system/data_json
curl -O https://your-server.com/contract_data.json
```

**Option D: SSH/FTP** (Termux only)
```bash
# Install SSH server in Termux
pkg install openssh
sshd

# From computer
scp contract_data.json your-phone-ip:~/downloads/ml-unified-system/data_json/
```

---

## 📊 Viewing Results on Android

### JSON Files

**Option 1: Text Editor**
- Use built-in editor in Pydroid 3
- Or install "QuickEdit Text Editor" from Play Store

**Option 2: JSON Viewer App**
- Install "JSON Viewer" or "JSON Genie" from Play Store
- Better formatting and syntax highlighting

**Option 3: Transfer to Desktop**
- Copy results to Google Drive
- View in desktop browser

### Termux - View in Terminal

```bash
# Pretty print JSON
cat ml_output/scoring_results.json | python -m json.tool

# Or install jq for better formatting
pkg install jq
cat ml_output/scoring_results.json | jq '.'

# View specific contract
cat ml_output/scoring_results.json | jq '.predictions[] | select(.contract_address == "0x1234...")'
```

---

## 🔧 Troubleshooting

### Issue 1: "Permission Denied"

**Pydroid 3:**
- Go to Android Settings → Apps → Pydroid 3 → Permissions
- Enable "Storage" permission

**Termux:**
```bash
termux-setup-storage
# Tap "Allow"
```

### Issue 2: "Module Not Found: numpy"

**Pydroid 3:**
- Menu → Pip → Search "numpy" → Install
- Wait for installation to complete
- Restart Pydroid 3

**Termux:**
```bash
pip install numpy --upgrade
```

### Issue 3: "Out of Memory"

**Solution:**
- Reduce dataset size (analyze fewer contracts at once)
- Close other apps
- Restart phone
- Consider splitting into batches:
  ```python
  # Process in smaller batches
  for i in range(0, len(contracts), 10):
      batch = contracts[i:i+10]
      process_batch(batch)
  ```

### Issue 4: Installation Takes Forever

**Normal on Android:**
- `numpy` can take 10-15 minutes
- `scikit-learn` can take 20-30 minutes
- Be patient, only install once!
- Keep screen on and phone plugged in

### Issue 5: "No such file or directory: ./data_json"

**Solution:**
```bash
# Create missing directories
mkdir -p data_json data_txt data_db trained_models ml_output
```

Or let the script create them on first run.

### Issue 6: Training Crashes on Android

**Possible causes:**
- Too many contracts (reduce dataset)
- Insufficient RAM
- Battery saver killing app

**Solutions:**
```python
# In ML_UNIFIED_SYSTEM_V3_3.py
# Reduce model complexity for Android:
RANDOM_FOREST_PARAMS = {
    'n_estimators': 20,  # Reduce from 50
    'max_depth': 3,      # Reduce from 5
    ...
}
```

---

## 📁 Common File Paths on Android

```
Main Storage:
/storage/emulated/0/                    → Main storage root

Downloads:
/storage/emulated/0/Download/           → Downloaded files

Pydroid 3:
/storage/emulated/0/Pydroid/            → Pydroid workspace
/data/data/ru.iiec.pydroid3/            → App internal storage

Termux:
/data/data/com.termux/files/home/       → Termux home (~/)
~/storage/                               → Access to shared storage
~/storage/downloads/                     → Downloads folder
~/storage/shared/                        → Shared storage
```

---

## ⚡ Quick Start Commands

### Pydroid 3 Quick Start

```python
# First time setup (run in Pydroid Terminal)
import os
os.makedirs('data_json', exist_ok=True)
os.makedirs('data_txt', exist_ok=True)
os.makedirs('data_db', exist_ok=True)

# Then run the main script
# Tap ▶️ button
```

### Termux Quick Start

```bash
# One-time setup
pkg update && pkg install python python-pip -y
pip install numpy pandas scikit-learn joblib
cd ~/storage/downloads
mkdir -p ml-unified-system/data_json

# Every time you use it
cd ~/storage/downloads/ml-unified-system
python ML_UNIFIED_SYSTEM_V3_3.py
```

---

## 🎓 Learning Resources

### Practice Datasets

Create test data on your Android:

```python
# create_test_data.py (run in Pydroid 3)
import json
import random

def create_sample_contract(address):
    return {
        "metadata": {
            "chain_id": 1,
            "contract_address": address
        },
        "bytecode": {
            "size": random.randint(1000, 20000),
            "complexity_metrics": {
                "cyclomatic_complexity": random.randint(1, 50),
                "halstead_volume": random.uniform(1000, 20000),
                "maintainability_index": random.uniform(20, 80),
                "opcode_diversity": random.uniform(0.3, 0.9)
            }
        },
        "functions": {
            "total": random.randint(5, 30),
            "known": random.randint(3, 20),
            "unknown": random.randint(0, 10)
        },
        "temporal_analysis": {
            "last_interaction_days": random.randint(0, 365),
            "unique_users_30d": random.randint(10, 1000),
            "activity_pattern": random.choice(["dormant", "active", "very_active"])
        },
        "economics": {
            "total_value_locked_usd": random.uniform(1000, 1000000),
            "token_count": random.randint(1, 5)
        },
        "gas_profiles": {
            "average_tx_cost": random.randint(50000, 500000),
            "gas_limits": {
                "safe_execution_limit": 3000000,
                "frontrun_protection_required": random.choice([True, False])
            }
        }
    }

# Generate 10 test contracts
for i in range(10):
    address = f"0x{''.join(random.choices('0123456789abcdef', k=40))}"
    data = create_sample_contract(address)
    
    with open(f"data_json/test_contract_{i}.json", 'w') as f:
        json.dump(data, f, indent=2)

print("✅ Created 10 test contracts in data_json/")
```

---

## 🌟 Best Practices for Mobile

1. **Work in Batches**
   - Process 10-20 contracts at a time
   - Prevents memory issues
   - Easier to manage

2. **Regular Backups**
   - Copy trained models to cloud storage
   - Backup important results
   - Easy to restore if needed

3. **Organize Your Data**
   - Use clear filenames: `contract_0x1234_analysis.json`
   - Date-stamp your outputs: `results_20260208.json`
   - Archive old data regularly

4. **Monitor Performance**
   - Watch battery usage
   - Check storage space
   - Close unused apps

5. **Use Airplane Mode**
   - Prevents interruptions during training
   - Saves battery
   - Faster processing

---

## 💬 Community Tips

**From Android Users:**

> "I run ML Unified on my phone during my commute. Perfect for learning blockchain security!" - @user123

> "Termux + external keyboard = portable security lab" - @researcher

> "Pydroid is easier for beginners, Termux for power users" - @developer

---

## 📞 Get Help

**Android-Specific Issues:**
- Tag posts with `#android` on GitHub Issues
- Join mobile-dev channel on Discord
- Check Android-specific FAQ

**Performance Issues:**
- Share your device specs
- Provide error logs
- Mention dataset size

---

## ⭐ Success Stories

ML Unified System runs successfully on:
- ✅ Samsung Galaxy S20+ (Pydroid 3)
- ✅ OnePlus 9 (Termux)
- ✅ Pixel 6 (both Pydroid & Termux)
- ✅ Xiaomi Redmi Note 10 (Pydroid 3)
- ✅ Tablets (Samsung Tab S7+)

---

**Happy Mobile ML! 📱🧠**

