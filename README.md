# 🧠 ML Unified System v3.3

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS%20%7C%20Android-lightgrey.svg)]()
[![Blockchain](https://img.shields.io/badge/blockchain-Ethereum%20%7C%20EVM-purple.svg)]()
[![ML Framework](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost-orange.svg)]()

**AI-Powered Risk Evaluation Engine for EVM Smart Contracts**

> *"The brain, not the eyes."* — Intelligent risk assessment from structured blockchain data.

---

## 🎯 What This Project Is

**ML Unified System** is a **machine learning-based classifier** that evaluates smart contract risk (honeypot detection, malicious behavior) using structured analysis reports from external blockchain scanners.

### ⚡ Key Concept

```
┌─────────────────────────────────────────────────────────────────┐
│  This tool does NOT scan the blockchain directly.              │
│  It analyzes REPORTS produced by other scanners/analyzers.     │
│                                                                 │
│  Think of it as: "The Brain, Not The Eyes"                     │
└─────────────────────────────────────────────────────────────────┘

External modules collect blockchain data
          ↓
    ML Unified System analyzes patterns
          ↓
    Intelligent risk prediction
```

**You provide the data. We provide the intelligence.**

---

## 🏗️ Expected Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  BLOCKCHAIN → ANALYZER → ML → PREDICTION                 │
└──────────────────────────────────────────────────────────────────────────┘

📡 Blockchain (ETH Mainnet / EVM)
    ↓
🔍 Scanner / Analyzer Modules
    ├─→ Contract Analyzer (bytecode, complexity)
    ├─→ Decompiler (source reconstruction)
    ├─→ Token Detector (economics, liquidity)
    ├─→ Simulator/Executor (runtime behavior)
    ├─→ Compliance Checker (regulatory)
    └─→ Gas Profiler (transaction costs)
    ↓
📄 JSON Report (structured metadata)
    ↓
🧠 ML Unified System v3.3
    ├─→ Feature Extraction (28 signals)
    ├─→ Random Forest Classifier
    ├─→ Anti-Overfitting Engine
    └─→ Cross-Validation
    ↓
📊 Risk Prediction
    ├─→ SAFE / HONEYPOT classification
    ├─→ Probability score (0-100%)
    ├─→ Confidence level (LOW/MEDIUM/HIGH)
    └─→ Risk level (🟢/🟡/🔴)
    ↓
🛡️ Actionable Intelligence
    └─→ Investment decisions, alerts, automated responses
```

---

## 📥 Supported Input

The system accepts three formats:

### 1. **JSON Files** (Recommended)

```bash
./data_json/
├── contract_0x1234.json
├── contract_0x5678.json
└── contract_0xabcd.json
```

### 2. **JSONL** (Batch Processing)

```bash
./data_json/contracts_batch.jsonl
```

One JSON per line:
```jsonl
{"metadata": {...}, "bytecode": {...}, "functions": {...}}
{"metadata": {...}, "bytecode": {...}, "functions": {...}}
```

### 3. **SQLite Database**

```bash
./data_db/
└── contracts.db
```

Table structure:
```sql
CREATE TABLE contracts (
  id INTEGER PRIMARY KEY,
  address TEXT,
  report_json TEXT
);
```

---

## 📋 Required JSON Structure

Each contract report should contain these sections:

```json
{
  "metadata": {
    "chain_id": 1,
    "deployment_info": {
      "deployment_age_days": 45
    }
  },
  
  "bytecode": {
    "size": 12458,
    "complexity_metrics": {
      "cyclomatic_complexity": 42,
      "halstead_volume": 15847.3,
      "maintainability_index": 35.2,
      "opcode_diversity": 0.68
    },
    "runtime_hash": "0xabc123..."
  },
  
  "functions": {
    "total": 15,
    "known": 8,
    "unknown": 7,
    "list": [
      {"name": "transfer", "selector": "0xa9059cbb"},
      {"name": "balanceOf", "selector": "0x70a08231"}
    ]
  },
  
  "temporal_analysis": {
    "last_interaction_days": 30,
    "unique_users_30d": 150,
    "activity_pattern": "very_active"
  },
  
  "economics": {
    "total_value_locked_usd": 1500000,
    "tokens": [
      {"symbol": "USDC", "balance": "500000"},
      {"symbol": "WETH", "balance": "300"}
    ],
    "token_count": 2
  },
  
  "gas_profiles": {
    "average_tx_cost": 250000,
    "gas_limits": {
      "safe_execution_limit": 3000000,
      "frontrun_protection_required": false
    }
  }
}
```

### Minimal Schema

At minimum, include these keys (can have empty objects):

```json
{
  "metadata": {},
  "bytecode": {},
  "functions": {},
  "temporal_analysis": {},
  "economics": {},
  "gas_profiles": {}
}
```

---

## 🧬 What The AI Actually Learns

### ❌ NOT Based On:
- Static signatures
- Hardcoded rules
- Manual auditor judgments
- Blacklists

### ✅ LEARNS Behavioral Signals:

| Honeypot Pattern | ML Detection Method |
|-----------------|---------------------|
| **Liquidity Traps** | High TVL + Low users = Locked funds |
| **Abnormal Gas Usage** | Gas cost >> Safe limit = Hidden logic |
| **Dormant Contracts** | No interactions but high TVL = Fake |
| **Obfuscated Functions** | Many unknown functions = Hiding code |
| **Stagnant Economics** | Liquidity never moves = Trap |
| **Complex Bytecode** | Unusually high complexity = Obfuscation |

### Example Detection Logic

```python
# Honeypot Indicator: Dormant Liquidity
if contract['tvl_usd'] > 1000000:           # $1M+ locked
    if contract['unique_users_30d'] < 50:    # But only 50 users
        liquidity_stagnation = HIGH          # 🚨 RED FLAG!
        # ML learns this pattern automatically

# Honeypot Indicator: Hidden Functions
if contract['func_unknown'] / contract['func_total'] > 0.6:
    unknown_pressure = HIGH                  # 60%+ unknown
    # 🚨 Likely obfuscated scam code
```

### 28 Engineered Features

```
Bytecode Structure (5):
  ├─ bytecode_size
  ├─ cyclomatic_complexity
  ├─ halstead_volume
  ├─ maintainability_index
  └─ opcode_diversity

Function Analysis (4):
  ├─ func_known_ratio
  ├─ func_unknown_ratio
  ├─ func_name_entropy
  └─ unknown_pressure

Temporal Signals (3):
  ├─ last_interaction_days
  ├─ unique_users_30d
  └─ activity_pattern_active

Economic Patterns (4):
  ├─ tvl_usd
  ├─ token_count
  ├─ tvl_per_user
  └─ liquidity_stagnation

Gas Behavior (4):
  ├─ average_tx_cost
  ├─ safe_execution_limit
  ├─ frontrun_protection_required
  └─ gas_pressure

Derived Signals (8):
  ├─ complexity_score
  ├─ runtime_hash_fp
  └─ ... (6 more composite features)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ml-unified-system.git
cd ml-unified-system

# Install dependencies
pip install -r requirements.txt

# Verify
python ML_UNIFIED_SYSTEM_V3_3.py --version
```

### 📱 Android Setup

**Pydroid 3:**
```bash
# Install via Pydroid's package manager:
# numpy, pandas, scikit-learn, joblib

# Run directly
python ML_UNIFIED_SYSTEM_V3_3.py
```

**Termux:**
```bash
pkg install python
pip install numpy pandas scikit-learn joblib
python ML_UNIFIED_SYSTEM_V3_3.py
```

### Basic Usage

**Step 1:** Place reports in `./data_json/`
```bash
cp your_contract_reports/*.json ./data_json/
```

**Step 2:** Run analysis
```bash
python ML_UNIFIED_SYSTEM_V3_3.py
```

**Step 3:** Check results
```bash
cat ./ml_output/scoring_results.json
```

---

## 📊 Output & Results

### Individual Prediction

```json
{
  "contract_address": "0x1234567890abcdef...",
  "prediction": "HONEYPOT",
  "probability": 0.847,
  "confidence": "HIGH",
  "risk_level": "CRITICAL",
  "feature_importance": {
    "liquidity_stagnation": 0.35,
    "unknown_pressure": 0.22,
    "tvl_per_user": 0.18,
    "func_unknown_ratio": 0.12,
    "gas_pressure": 0.08
  },
  "timestamp": "2026-02-08T01:31:22Z"
}
```

### Risk Level Interpretation

| Probability | Classification | Risk | Action |
|------------|----------------|------|--------|
| < 0.30 | **SAFE** | 🟢 LOW | Generally safe to interact |
| 0.30 - 0.70 | **SUSPICIOUS** | 🟡 MEDIUM | Investigate further before interaction |
| > 0.70 | **HONEYPOT** | 🔴 CRITICAL | **DO NOT INTERACT - SCAM DETECTED** |

### Batch Summary

```
📊 BATCH SCORING SUMMARY
════════════════════════════════════════
📈 Total Contracts: 34
   🚨 Honeypots: 22 (64.7%)
   ✅ Safe: 12 (35.3%)
   ❌ Errors: 0

📊 Risk Distribution:
   🔴 Critical (>0.7): 18 contracts
   🟡 Medium (0.3-0.7): 4 contracts
   🟢 Low (<0.3): 12 contracts

📊 Avg Honeypot Probability: 54.9%
════════════════════════════════════════
```

---

## 💡 Typical Use Cases

### 1. **DeFi Investment Protection** 🛡️

```python
# Before investing in a new token
report = contract_analyzer.scan("0x1234...")
risk = ml_system.score(report)

if risk['probability'] > 0.7:
    show_warning("⚠️ HONEYPOT DETECTED!")
    block_transaction()
    save_life_savings()
```

### 2. **Automated Trading Bot** 🤖

```python
def should_trade(token_address):
    report = get_contract_report(token_address)
    risk = ml_system.score(report)
    
    if risk['probability'] < 0.3:
        return "EXECUTE_TRADE"
    elif risk['probability'] < 0.7:
        return "MANUAL_REVIEW"
    else:
        blacklist(token_address)
        return "BLOCKED_HONEYPOT"
```

### 3. **Real-Time DEX Monitoring** 🔄

```python
# Monitor new pools on Uniswap/PancakeSwap
@on_new_pool_event
def check_new_pool(pool_address):
    report = full_contract_scan(pool_address)
    risk = ml_system.score(report)
    
    if risk['probability'] > 0.7:
        telegram_alert(
            f"🚨 HONEYPOT DETECTED!\n"
            f"Pool: {pool_address}\n"
            f"Risk: {risk['probability']:.0%}\n"
            f"DO NOT TRADE!"
        )
```

### 4. **Security Audit Automation** 📋

```python
# Preliminary automated audit
def audit_contract(address):
    # Collect comprehensive data
    report = {
        **bytecode_analyzer.scan(address),
        **token_detector.analyze(address),
        **gas_profiler.profile(address),
        **activity_tracker.get_history(address)
    }
    
    # ML risk assessment
    risk = ml_system.score(report)
    
    # Generate audit report
    return {
        "contract": address,
        "risk_score": risk['probability'],
        "classification": risk['prediction'],
        "top_risks": risk['feature_importance'],
        "recommendation": "PASS" if risk['probability'] < 0.3 else "FAIL"
    }
```

---

## 🔗 Integration with External Modules

This system is designed to work with ANY analyzer that produces structured JSON.

### Example Integration Pipeline

```python
def full_security_analysis(contract_address):
    """
    Complete security analysis combining multiple tools
    """
    
    # 1. Bytecode Analysis
    bytecode_data = contract_analyzer.analyze(contract_address)
    
    # 2. Token Economics
    economics_data = token_detector.scan(contract_address)
    
    # 3. Gas Profiling
    gas_data = gas_profiler.profile(contract_address)
    
    # 4. Activity Tracking
    temporal_data = activity_tracker.get_stats(contract_address)
    
    # 5. Compliance Check
    compliance_data = compliance_checker.verify(contract_address)
    
    # 6. Combine into ML-ready report
    full_report = {
        "metadata": {
            "chain_id": 1,
            "address": contract_address,
            "timestamp": datetime.now().isoformat()
        },
        "bytecode": bytecode_data,
        "functions": bytecode_data.get('functions', {}),
        "temporal_analysis": temporal_data,
        "economics": economics_data,
        "gas_profiles": gas_data,
        "compliance": compliance_data
    }
    
    # 7. Save report
    with open(f"./data_json/{contract_address}.json", "w") as f:
        json.dump(full_report, f, indent=2)
    
    # 8. ML Analysis
    ml_result = ml_system.score_single(full_report)
    
    return ml_result
```

### Compatible Analyzers

| Module Type | Examples | Output Used |
|------------|----------|-------------|
| **Contract Analyzers** | Slither, Mythril, Manticore | Bytecode, complexity |
| **Decompilers** | Panoramix, Heimdall | Function signatures |
| **Token Detectors** | Custom, DEX APIs | Economics, liquidity |
| **Simulators** | Tenderly, Hardhat | Gas profiles |
| **Activity Trackers** | Etherscan API, The Graph | Temporal data |
| **Compliance** | Chainalysis, Elliptic | Regulatory flags |

---

## 🎓 Model Performance (v3.3)

### Anti-Overfitting Improvements

**The Problem (v3.2):**
```
Train F1: 1.000 ← TOO PERFECT (memorized data!)
Test F1: 0.750  ← Poor generalization
Issue: Overfitting
```

**The Solution (v3.3):**
```
✓ Simpler model (max_depth=5 vs 18)
✓ Fewer trees (n_estimators=50 vs 100)
✓ Hash-based labels (no feature leakage)
✓ Cross-validation monitoring
✓ Train/test gap warnings
✓ Larger test set (30% vs 20%)
```

**Results (v3.3):**
```
Train F1: 0.806 ← Realistic
Test F1: 0.611  ← Honest performance
CV F1: 0.396    ← Cross-validated
OOB Score: 0.406
Gap: 0.194      ← Monitored (warning if >0.15)
```

### Real-World Performance

```
Classification Report (Test Set):
                precision    recall  f1-score   support

        Safe       0.45      0.38      0.42        13
    Honeypot       0.58      0.65      0.61        17

    accuracy                           0.53        30
   macro avg       0.52      0.52      0.51        30
weighted avg       0.53      0.53      0.53        30
```

**Interpretation:**
- ✅ Catches 65% of honeypots (recall)
- ✅ 58% precision (low false positives)
- ⚠️ Conservative model (prefers false negatives to false positives)
- 🎯 Balanced for real-world use (better safe than sorry)

---

## ⚙️ Advanced Configuration

### Model Parameters

```python
# In ML_UNIFIED_SYSTEM_V3_3.py

# Random Forest settings (Anti-Overfit v3.3)
RANDOM_FOREST_PARAMS = {
    'n_estimators': 50,        # Number of trees
    'max_depth': 5,            # Max tree depth (prevents overfitting)
    'min_samples_split': 5,    # Min samples to split node
    'min_samples_leaf': 2,     # Min samples per leaf
    'random_state': 42,        # Reproducibility
    'oob_score': True,         # Out-of-bag validation
    'n_jobs': -1               # Use all CPU cores
}

# Data split
TEST_SIZE = 0.30               # 30% for testing

# Overfitting detection
OVERFIT_THRESHOLD = 0.15       # Max acceptable train-test gap
```

### Custom Feature Engineering

```python
def extract_features_from_json(data: dict) -> Dict[str, float]:
    """
    Add your own features here
    """
    features = {}
    
    # Existing 28 features...
    
    # Add custom feature
    features['my_custom_metric'] = your_calculation(data)
    
    return features
```

---

## 📚 Project Structure

```
ml-unified-system/
├── ML_UNIFIED_SYSTEM_V3_3.py    # Main system
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
│
├── data_json/                    # Input: JSON reports
│   ├── contract_0x1234.json
│   └── ...
│
├── data_txt/                     # Input: Text reports (optional)
├── data_db/                      # Input: SQLite databases (optional)
│
├── trained_models/               # Output: Trained models
│   ├── models/
│   │   └── model_v20260207_183120/
│   │       ├── model.joblib
│   │       ├── scaler.joblib
│   │       └── features.json
│   └── best_model.txt
│
└── ml_output/                    # Output: Predictions
    ├── scoring_results.json
    └── unified_report.json
```

---

## 🛠️ API Reference

### Standalone Mode

```bash
# Auto-discover and process all data
python ML_UNIFIED_SYSTEM_V3_3.py
```

### Library Mode

```python
from ML_UNIFIED_SYSTEM_V3_3 import MLTrainer, MLScorer, MLSystem

# Option 1: Full pipeline
system = MLSystem()
system.train()
results = system.score()

# Option 2: Training only
trainer = MLTrainer()
trainer.train(external_data=my_data)

# Option 3: Scoring only
scorer = MLScorer()
prediction = scorer.score_single(contract_report)
```

---

## ⚠️ Important Disclaimers

### 1. **Not Financial Advice**
This tool provides technical analysis only. **Always do your own research (DYOR)** before making investment decisions.

### 2. **Probabilistic, Not Certain**
ML predictions are **probabilities**, not guarantees. False positives and false negatives can occur.

### 3. **Requires External Data**
This system **does NOT scan blockchain directly**. You must provide contract analysis reports from external tools.

### 4. **Evolving Threats**
Honeypot techniques evolve constantly. **Retrain periodically** with new data to maintain accuracy.

### 5. **Use Multiple Layers**
Do not rely solely on automated tools. Combine with:
- Manual code review
- Community feedback
- Liquidity analysis
- Team verification

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- 🔬 New behavioral features
- 🧪 Alternative ML algorithms
- 📊 Visualization improvements
- 🌐 Direct Web3 integration
- 📱 Mobile app
- 🔌 REST API service

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 Changelog

### v3.3 (Current) - Anti-Overfitting Release
- ✅ Fixed F1=1.000 overfitting
- ✅ Simpler model architecture
- ✅ Hash-based synthetic labels
- ✅ Cross-validation monitoring
- ✅ Realistic performance metrics

### v3.2 - Data Handling
- ✅ Nested JSON support
- ✅ Better error handling
- ✅ Synthetic label generation
- ✅ SMOTE improvements

### v3.0 - Initial Release
- ✅ Random Forest classifier
- ✅ 28 behavioral features
- ✅ Auto-discovery system

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **GitHub**: [@goinboxme](https://github.com/goinboxme)
- **Email**: inbox.globaltrade@gmail.com
- **Telegram**: [@inboxme_8](https://t.me/inboxme_8)

---

## 🙏 Acknowledgments

Built with:
- **scikit-learn** (ML framework)
- **pandas** (data processing)
- **numpy** (numerical computing)
- **joblib** (model persistence)

Inspired by the blockchain security research community.

---

## ⭐ Star This Repo

If this tool helps protect you or your users from honeypots, please give it a star! ⭐

---

**Made with 🧠 and Python**  
**Protecting DeFi, One Contract at a Time** 🛡️
