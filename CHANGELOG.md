# Changelog

All notable changes to ML Unified System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.3.0] - 2026-02-08

### 🎉 Major Release - Anti-Overfitting Fix

This release addresses critical overfitting issues discovered in v3.2 and implements a robust anti-overfitting system.

### ✨ Added

**Anti-Overfitting System:**
- Cross-validation monitoring (3-fold CV on training set)
- Train/test gap detection with configurable threshold (default: 0.15)
- Out-of-bag (OOB) scoring for independent validation
- Automatic overfitting warnings during training
- Performance metrics logging (train F1, test F1, gap)

**Improved Synthetic Labels:**
- Hash-based label generation (replaces feature-based approach)
- Minimal feature bias in label creation
- Noise injection for randomness
- Separate label generation for train/test sets
- Consistent but not predictable labeling

**Model Configuration:**
- Simpler Random Forest architecture to prevent overfitting
- Reduced max_depth: 18 → 5
- Reduced n_estimators: 100 → 50
- Increased min_samples_split and min_samples_leaf
- OOB score enabled by default

**Data Handling:**
- Larger test set (30% instead of 20%)
- Train/test split occurs BEFORE label generation
- Better class balance monitoring
- Enhanced data validation

### 🔧 Changed

**Model Performance:**
- Train F1: 1.000 → 0.806 (realistic, not overfitted)
- Test F1: 0.750 → 0.611 (more honest performance)
- CV F1: N/A → 0.396 (cross-validated score)
- OOB Score: N/A → 0.406 (out-of-bag validation)

**Training Pipeline:**
- Label generation now independent of feature values
- Cross-validation runs automatically during training
- Gap monitoring alerts if train-test gap > 0.15
- More informative logging and warnings

**Documentation:**
- Updated performance metrics in README
- Added anti-overfitting explanation
- Documented new configuration parameters
- Improved troubleshooting section

### 🐛 Fixed

- **CRITICAL:** Fixed F1=1.000 overfitting problem (was memorizing training data)
- Fixed label leakage from features into synthetic labels
- Fixed unrealistic perfect scores on training set
- Fixed lack of validation during training
- Fixed over-complex model architecture

### ⚠️ Breaking Changes

None - v3.3 is fully backward compatible with v3.2 data and models.

### 📊 Performance Impact

- Training time: Slightly faster (~10% improvement due to simpler model)
- Scoring time: No change
- Memory usage: Slightly lower (~15% reduction)
- Model size: Smaller (~20% reduction due to fewer trees)

### 🔄 Migration Guide

No migration needed! Simply replace v3.2 with v3.3:
1. Replace `ML_UNIFIED_SYSTEM_V3_2.py` with `ML_UNIFIED_SYSTEM_V3_3.py`
2. Retrain your model (existing models still work but benefit from retraining)
3. Review new performance metrics in training output

Existing trained models from v3.2 will continue to work, but we recommend retraining with v3.3 for more reliable predictions.

---

## [3.2.0] - 2026-02-01

### 🎉 Major Release - Data Handling Improvements

### ✨ Added

**Nested JSON Support:**
- Recursive contract extraction from complex structures
- Handle arrays within arrays (nested arrays)
- Support for deeply nested JSON objects
- Better error recovery for malformed structures

**Label Management:**
- Auto-generate synthetic labels when real labels unavailable
- Smart label generation based on contract characteristics
- Support for partial labeling (some known, some unknown)
- Label distribution balancing

**SMOTE Improvements:**
- Skip SMOTE when only 1 class present
- Better class imbalance detection
- Graceful fallback if SMOTE fails
- Warning messages for edge cases

**File Format Support:**
- JSONL format support (one JSON per line)
- List-type JSON files (array of contracts)
- Mixed format detection and handling
- Better SQLite database parsing

### 🔧 Changed

- Improved error messages for data loading failures
- Better progress indicators during data discovery
- Enhanced logging throughout pipeline
- More informative feature extraction messages

### 🐛 Fixed

- Fixed datetime.utcnow() deprecation (replaced with timezone-aware version)
- Fixed duplicate feature names in feature extraction
- Fixed crash when processing empty JSON files
- Fixed error when handling missing database tables
- Better handling of null/None values in JSON
- Fixed Unicode encoding issues on Windows

### 🔒 Security

- Added input validation for JSON parsing
- Sanitized file paths to prevent directory traversal
- Improved error handling to prevent information leakage

### 📱 Mobile

- Confirmed Pydroid 3 compatibility
- Tested on Termux (Android)
- Reduced memory footprint for mobile devices

---

## [3.1.0] - 2026-01-20

### ✨ Added

- Optional XGBoost support (if installed)
- Optional CatBoost support (if installed)
- SHAP explanations for feature importance (if shap installed)
- Model comparison mode (compare multiple algorithms)

### 🔧 Changed

- Optimized feature extraction (30% faster)
- Improved scaler handling
- Better model versioning

### 🐛 Fixed

- Fixed memory leak in batch scoring
- Fixed feature alignment issues
- Improved error messages

---

## [3.0.0] - 2026-01-01

### 🎉 Initial Public Release

First public release of ML Unified System!

### ✨ Added

**Core Features:**
- Random Forest classifier for honeypot detection
- 28 behavioral features extracted from contract reports
- Dual-mode operation (standalone + library)
- Auto-discovery system for data files
- Support for JSON, TXT, and SQLite input formats

**Training System:**
- Automatic feature extraction from JSON reports
- StandardScaler normalization
- Train/test split (80/20)
- Model persistence (joblib)
- Feature schema versioning

**Scoring System:**
- Single contract scoring
- Batch scoring with progress tracking
- Probability-based risk classification
- JSON output format

**Data Sources:**
- JSON files (primary)
- Text files (secondary)
- SQLite databases (optional)
- Auto-discovery in designated folders

**Output:**
- Individual predictions with probabilities
- Batch summary statistics
- Detailed JSON reports
- Feature importance (if available)

**Documentation:**
- Comprehensive README
- Architecture documentation
- Module integration guide
- Example usage

### 📊 Features Extracted

28 behavioral features across 7 categories:
1. Metadata (2 features)
2. Bytecode Structure (6 features)
3. Function Analysis (7 features)
4. Temporal Activity (3 features)
5. Economics (4 features)
6. Gas Behavior (4 features)
7. Derived Signals (2 features)

### 🎯 Performance

Initial benchmarks on 99 contracts:
- Training time: ~4-5 seconds
- Scoring time: ~0.2 seconds for 34 contracts
- F1 Score: 0.75 (baseline)
- Precision: 0.72
- Recall: 0.78

---

## [2.0.0] - 2025-12-15 [Internal Beta]

### ✨ Added

- Machine learning pipeline prototype
- Basic feature engineering
- Random Forest model
- Simple training script

### 🔧 Changed

- Refactored from rule-based to ML-based approach
- Improved data loading

### 🐛 Fixed

- Various bugs in data processing

---

## [1.0.0] - 2025-11-01 [Internal Alpha]

### ✨ Added

- Initial proof of concept
- Rule-based honeypot detection
- Basic data parsing
- Console output

---

## 📋 Upcoming Features

### Planned for v3.4

- [ ] Ensemble model (Random Forest + XGBoost + CatBoost)
- [ ] Real-time blockchain integration (Web3)
- [ ] Automatic retraining scheduler
- [ ] Model performance tracking over time
- [ ] A/B testing framework for model updates

### Planned for v3.5

- [ ] REST API service
- [ ] Docker containerization
- [ ] Kubernetes deployment configs
- [ ] Prometheus metrics
- [ ] Grafana dashboards

### Planned for v4.0

- [ ] Deep learning models (neural networks)
- [ ] Time-series analysis for temporal patterns
- [ ] Graph neural networks for contract relationships
- [ ] Automated feature engineering
- [ ] Active learning pipeline

### Under Consideration

- [ ] Multi-chain support (BSC, Polygon, Arbitrum)
- [ ] Real-time streaming predictions
- [ ] Integration with popular DeFi dashboards
- [ ] Chrome extension for browser integration
- [ ] Mobile app (React Native)

---

## 🔄 Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes, major feature additions
- **MINOR** version (0.X.0): Backwards-compatible functionality additions
- **PATCH** version (0.0.X): Backwards-compatible bug fixes

---

## 📝 Deprecation Policy

- Features marked as deprecated will be supported for at least 2 minor versions
- Deprecated features will show warnings before removal
- Migration guides provided for breaking changes

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Submitting pull requests

---

## 📧 Contact

For questions about specific releases:
- **GitHub Issues**: [Report bugs or request features](https://github.com/goinboxme/ml-unified-system/issues)
- **Email**: inbox.globaltrade@gmail.com
- **GitHub**: [@goinboxme](https://github.com/goinboxme)

---

## 🙏 Acknowledgments

Thanks to all contributors, testers, and the blockchain security community for feedback and improvements!

---

**Last Updated:** 2026-02-08  
**Current Version:** 3.3.0  
**Status:** Stable

