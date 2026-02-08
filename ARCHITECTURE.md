# 🏗️ Architecture Guide

Complete technical architecture of ML Unified System v3.3

---

## 📐 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ML UNIFIED SYSTEM v3.3 ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA INGESTION                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  • Auto-Discovery Engine                                                │
│  • Multi-Format Parser (JSON, JSONL, SQLite)                           │
│  • Contract Extraction (handles nested structures)                      │
│  • Data Validation & Error Handling                                     │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: FEATURE ENGINEERING                                           │
├─────────────────────────────────────────────────────────────────────────┤
│  • JSON → Feature Vector (28 dimensions)                                │
│  • Leak-Free Extraction (no analyzer bias)                             │
│  • Behavioral Signal Detection                                          │
│  • Feature Normalization (StandardScaler)                               │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: LABEL GENERATION                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  • Real Labels (from DB if available)                                   │
│  • Synthetic Labels (hash-based, v3.3)                                  │
│  • Train/Test Split (70/30)                                            │
│  • Class Balance Check                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: MODEL TRAINING                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  • Random Forest (primary, anti-overfit config)                         │
│  • Cross-Validation (3-fold)                                           │
│  • Out-of-Bag Validation                                               │
│  • Overfitting Detection                                               │
│  • Performance Metrics (F1, Precision, Recall)                         │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: MODEL PERSISTENCE                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  • Model Serialization (joblib)                                         │
│  • Scaler Serialization                                                 │
│  • Feature Schema Storage (JSON)                                        │
│  • Version Management                                                    │
│  • Best Model Pointer                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: INFERENCE ENGINE                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  • Model Loading (auto-discover best model)                            │
│  • Single Contract Scoring                                              │
│  • Batch Scoring                                                         │
│  • Probability → Risk Classification                                    │
│  • Feature Importance Analysis                                          │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 7: OUTPUT GENERATION                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  • JSON Reports (structured predictions)                                │
│  • Batch Summaries                                                       │
│  • Performance Reports                                                   │
│  • Logging & Alerts                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Training Pipeline

```
1. DATA DISCOVERY
   ├─ Scan ./data_json/ for JSON files
   ├─ Scan ./data_txt/ for text reports
   ├─ Scan ./data_db/ for SQLite databases
   └─ Extract contracts from all sources
        ↓
2. FEATURE EXTRACTION
   ├─ Parse JSON structure
   ├─ Extract 28 behavioral features per contract
   ├─ Handle missing/malformed data
   └─ Create feature matrix X (n_samples × 28)
        ↓
3. LABEL ASSIGNMENT
   ├─ Check for real labels in databases
   ├─ Generate synthetic labels if needed (v3.3 hash-based)
   ├─ Split train/test BEFORE labeling (prevents leakage)
   └─ Create label vector y (n_samples × 1)
        ↓
4. PREPROCESSING
   ├─ Feature normalization (StandardScaler)
   ├─ Fit scaler on training data only
   └─ Transform both train and test sets
        ↓
5. MODEL TRAINING
   ├─ Initialize Random Forest (anti-overfit params)
   ├─ Fit on training data (X_train, y_train)
   ├─ Cross-validation (3-fold on train set)
   ├─ Out-of-bag validation
   └─ Compute train/test gap
        ↓
6. OVERFITTING DETECTION
   ├─ Compare train F1 vs test F1
   ├─ Check gap > threshold (0.15)
   ├─ Log warning if overfitting detected
   └─ Provide recommendations
        ↓
7. MODEL PERSISTENCE
   ├─ Save model (model.joblib)
   ├─ Save scaler (scaler.joblib)
   ├─ Save feature names (features.json)
   ├─ Save metadata (training date, version)
   └─ Update best model pointer
```

### Scoring Pipeline

```
1. MODEL LOADING
   ├─ Read best model pointer
   ├─ Load model.joblib
   ├─ Load scaler.joblib
   └─ Load features.json
        ↓
2. DATA DISCOVERY
   ├─ Scan ./data_json/ for new contracts
   ├─ Parse JSON files
   └─ Extract contracts (handles nested arrays)
        ↓
3. FEATURE EXTRACTION
   ├─ Apply SAME feature extraction as training
   ├─ Use SAME 28 features
   ├─ Handle missing data gracefully
   └─ Create feature matrix X_new
        ↓
4. PREPROCESSING
   ├─ Apply SAME scaler (fitted during training)
   ├─ Transform features to same scale
   └─ Ensure feature order matches training
        ↓
5. PREDICTION
   ├─ model.predict_proba(X_new_scaled)
   ├─ Get probability scores [P(safe), P(honeypot)]
   ├─ Extract honeypot probability
   └─ Apply risk thresholds
        ↓
6. CLASSIFICATION
   ├─ prob < 0.3 → SAFE (🟢)
   ├─ 0.3 ≤ prob < 0.7 → SUSPICIOUS (🟡)
   └─ prob ≥ 0.7 → HONEYPOT (🔴)
        ↓
7. OUTPUT GENERATION
   ├─ Individual predictions (JSON)
   ├─ Batch summary statistics
   ├─ Feature importance (if SHAP available)
   └─ Save to ./ml_output/
```

---

## 🧬 Feature Engineering Pipeline

### Feature Extraction Process

```python
Input: JSON contract report
    ↓
┌──────────────────────────────────────────────────┐
│  SECTION 1: METADATA (2 features)               │
│  ├─ chain_id                                    │
│  └─ deployment_age_days                         │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│  SECTION 2: BYTECODE STRUCTURE (6 features)     │
│  ├─ bytecode_size                               │
│  ├─ cyclomatic_complexity                       │
│  ├─ halstead_volume                             │
│  ├─ maintainability_index                       │
│  ├─ opcode_diversity                            │
│  ├─ runtime_hash_len                            │
│  └─ runtime_hash_fp (fingerprint)               │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│  SECTION 3: FUNCTION ANALYSIS (7 features)      │
│  ├─ func_total                                  │
│  ├─ func_known                                  │
│  ├─ func_unknown                                │
│  ├─ func_known_ratio                            │
│  ├─ func_unknown_ratio                          │
│  ├─ func_name_entropy                           │
│  └─ unknown_pressure (derived)                  │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│  SECTION 4: TEMPORAL ACTIVITY (3 features)      │
│  ├─ last_interaction_days                       │
│  ├─ unique_users_30d                            │
│  └─ activity_pattern_active (binary)            │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│  SECTION 5: ECONOMICS (4 features)              │
│  ├─ tvl_usd                                     │
│  ├─ token_count                                 │
│  ├─ tvl_per_user (derived)                     │
│  └─ liquidity_stagnation (derived)              │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│  SECTION 6: GAS BEHAVIOR (4 features)           │
│  ├─ average_tx_cost                             │
│  ├─ safe_execution_limit                        │
│  ├─ frontrun_protection_required (binary)       │
│  └─ gas_pressure (derived)                      │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│  SECTION 7: DERIVED SIGNALS (2 features)        │
│  ├─ complexity_score (composite)                │
│  └─ (others computed in above sections)         │
└──────────────────────────────────────────────────┘
    ↓
Output: Feature vector (28 dimensions)
```

### Critical Features (High Impact)

**Top 5 Honeypot Indicators:**

1. **liquidity_stagnation** (35% importance)
   ```
   = TVL / (unique_users_30d + 1)
   
   High value = lots of locked money, few users
   Classic honeypot signature!
   ```

2. **unknown_pressure** (22% importance)
   ```
   = func_unknown / (func_total + 1)
   
   High value = many hidden functions
   Indicates obfuscated/malicious code
   ```

3. **tvl_per_user** (18% importance)
   ```
   = tvl_usd / unique_users_30d
   
   Very high value = abnormal concentration
   Real DEX has distributed liquidity
   ```

4. **func_unknown_ratio** (12% importance)
   ```
   = func_unknown / func_total
   
   Similar to unknown_pressure
   Measures code transparency
   ```

5. **gas_pressure** (8% importance)
   ```
   = average_tx_cost / safe_execution_limit
   
   High value = near gas limit
   May indicate hidden computation
   ```

---

## 🤖 Model Architecture

### Random Forest Classifier (v3.3 Anti-Overfit)

```
Model Configuration:
┌────────────────────────────────────────────────┐
│  Algorithm: Random Forest                     │
│  Purpose: Binary Classification               │
│  Classes: [0=Safe, 1=Honeypot]               │
├────────────────────────────────────────────────┤
│  Hyperparameters (Anti-Overfit v3.3):        │
│  ├─ n_estimators: 50 trees                   │
│  ├─ max_depth: 5 levels                      │
│  ├─ min_samples_split: 5                     │
│  ├─ min_samples_leaf: 2                      │
│  ├─ criterion: gini                          │
│  ├─ bootstrap: True                          │
│  ├─ oob_score: True                          │
│  └─ random_state: 42                         │
└────────────────────────────────────────────────┘
```

### Training Strategy

```
┌─────────────────────────────────────────────────────┐
│  ANTI-OVERFITTING MECHANISMS (v3.3)                │
├─────────────────────────────────────────────────────┤
│  1. Simpler Architecture                            │
│     ├─ Reduced max_depth (18 → 5)                  │
│     ├─ Fewer trees (100 → 50)                      │
│     └─ Higher min_samples constraints               │
│                                                     │
│  2. Better Data Handling                            │
│     ├─ Train/test split BEFORE labeling            │
│     ├─ Larger test set (20% → 30%)                 │
│     └─ Separate label generation per set           │
│                                                     │
│  3. Synthetic Labels (v3.3 Improved)               │
│     ├─ Hash-based (not feature-based)              │
│     ├─ Minimal feature bias                        │
│     ├─ Noise injection for randomness              │
│     └─ Consistent but not predictable              │
│                                                     │
│  4. Validation Strategy                            │
│     ├─ Cross-validation (3-fold)                   │
│     ├─ Out-of-bag scoring                          │
│     ├─ Train/test gap monitoring                   │
│     └─ Warning if gap > 0.15                       │
└─────────────────────────────────────────────────────┘
```

### Performance Metrics

```
┌─────────────────────────────────────────────────────┐
│  EVALUATION METRICS                                 │
├─────────────────────────────────────────────────────┤
│  Primary: F1 Score                                  │
│  ├─ Balances precision and recall                  │
│  ├─ Important for imbalanced classes               │
│  └─ Formula: 2 * (P * R) / (P + R)                 │
│                                                     │
│  Secondary: Precision                               │
│  ├─ TP / (TP + FP)                                 │
│  ├─ "Of predicted honeypots, how many are real?"   │
│  └─ Minimizes false alarms                         │
│                                                     │
│  Tertiary: Recall                                   │
│  ├─ TP / (TP + FN)                                 │
│  ├─ "Of real honeypots, how many did we catch?"    │
│  └─ Minimizes missed scams                         │
│                                                     │
│  Validation: OOB Score                              │
│  ├─ Out-of-bag accuracy                            │
│  ├─ Independent validation                         │
│  └─ Detects overfitting                            │
│                                                     │
│  Overfitting Check: Train-Test Gap                 │
│  ├─ Gap = Train_F1 - Test_F1                       │
│  ├─ Threshold: 0.15                                │
│  └─ Warning if exceeded                            │
└─────────────────────────────────────────────────────┘
```

---

## 💾 File System Architecture

### Directory Structure

```
ml-unified-system/
│
├── ML_UNIFIED_SYSTEM_V3_3.py          # Main system (1462 lines)
│
├── requirements.txt                    # Dependencies
├── README.md                           # Main documentation
├── ARCHITECTURE.md                     # This file
├── LICENSE                             # MIT License
│
├── data_json/                          # INPUT: JSON reports
│   ├── contract_0x1234.json
│   ├── contract_0x5678.json
│   ├── batch_analysis.jsonl           # JSONL format
│   └── ...
│
├── data_txt/                           # INPUT: Text reports (optional)
│   └── *.txt
│
├── data_db/                            # INPUT: SQLite databases (optional)
│   └── *.db
│
├── trained_models/                     # OUTPUT: Models
│   ├── models/
│   │   ├── model_v20260207_183120/
│   │   │   ├── model.joblib           # Random Forest model
│   │   │   ├── scaler.joblib          # StandardScaler
│   │   │   └── features.json          # Feature schema
│   │   │
│   │   └── model_v20260208_091530/    # Another version
│   │       └── ...
│   │
│   └── best_model.txt                  # Pointer to best model
│
└── ml_output/                          # OUTPUT: Predictions
    ├── scoring_results.json            # Individual predictions
    ├── unified_report.json             # Complete analysis
    └── batch_summary.txt               # Human-readable summary
```

### File Formats

**Model Files (joblib):**
```python
# model.joblib structure
{
    'estimator': RandomForestClassifier,
    'n_features_in_': 28,
    'classes_': array([0, 1]),  # Safe, Honeypot
    'n_estimators': 50,
    'max_depth': 5,
    # ... other sklearn attributes
}
```

**Scaler Files (joblib):**
```python
# scaler.joblib structure
{
    'mean_': array([...]),      # Feature means
    'scale_': array([...]),     # Feature scales
    'n_features_in_': 28,
    'feature_names_in_': array([...])
}
```

**Features Schema (JSON):**
```json
{
  "version": "3.3",
  "feature_count": 28,
  "feature_names": [
    "chain_id",
    "deployment_age_days",
    "bytecode_size",
    ...
  ],
  "feature_types": {
    "chain_id": "numeric",
    "deployment_age_days": "numeric",
    ...
  }
}
```

---

## 🔌 API Design

### Class: MLTrainer

```python
class MLTrainer:
    """
    Handles model training pipeline
    """
    
    def __init__(
        self,
        auto_discover: bool = True,
        use_synthetic_labels: bool = True
    ):
        """
        Initialize trainer
        
        Args:
            auto_discover: Auto-discover data files
            use_synthetic_labels: Generate labels if missing
        """
    
    def train(
        self,
        external_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Train ML model
        
        Args:
            external_data: Optional pre-loaded data
            
        Returns:
            Training results and metrics
        """
```

### Class: MLScorer

```python
class MLScorer:
    """
    Handles model inference
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None
    ):
        """
        Initialize scorer
        
        Args:
            model_path: Path to model (auto-detect if None)
        """
    
    def score_single(
        self,
        contract_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score single contract
        
        Args:
            contract_data: JSON contract report
            
        Returns:
            Prediction result
        """
    
    def score_batch(
        self,
        contracts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score multiple contracts
        
        Args:
            contracts: List of contract reports
            
        Returns:
            List of predictions
        """
```

### Class: MLSystem

```python
class MLSystem:
    """
    Unified interface (Trainer + Scorer)
    """
    
    def __init__(self):
        """Initialize unified system"""
    
    def train(self) -> Dict[str, Any]:
        """Run training pipeline"""
    
    def score(self) -> Dict[str, Any]:
        """Run scoring pipeline"""
```

---

## 🔐 Security Considerations

### Data Privacy
- ✅ No external API calls (fully offline)
- ✅ Local processing only
- ✅ No data transmission
- ✅ No telemetry

### Model Security
- ✅ Deterministic training (random_state=42)
- ✅ Version-controlled models
- ✅ Integrity checks (file hashes)
- ⚠️ No encryption (add if needed)

### Input Validation
- ✅ JSON schema validation
- ✅ Type checking
- ✅ Range validation for features
- ✅ Malformed data handling

---

## ⚡ Performance Optimization

### Training Performance
```
Typical Training Time (99 contracts):
├─ Data Loading: ~2 seconds
├─ Feature Extraction: ~1 second
├─ Model Training: ~0.5 seconds
├─ Cross-Validation: ~1 second
└─ Total: ~4.5 seconds
```

### Scoring Performance
```
Typical Scoring Time (34 contracts):
├─ Model Loading: ~0.05 seconds
├─ Data Loading: ~0.1 seconds
├─ Feature Extraction: ~0.05 seconds
├─ Prediction: ~0.01 seconds
└─ Total: ~0.2 seconds

Throughput: ~170 contracts/second
```

### Optimization Tips
1. **Batch Processing**: Score multiple contracts together
2. **Model Caching**: Keep model loaded in memory
3. **Feature Pre-computation**: Cache extracted features
4. **Parallel Processing**: Use `n_jobs=-1` in Random Forest

---

## 🧪 Testing Strategy

### Unit Tests
```python
# test_feature_extraction.py
def test_extract_features_minimal():
    """Test with minimal JSON"""
    
def test_extract_features_full():
    """Test with complete JSON"""
    
def test_extract_features_malformed():
    """Test error handling"""
```

### Integration Tests
```python
# test_pipeline.py
def test_full_training_pipeline():
    """End-to-end training test"""
    
def test_full_scoring_pipeline():
    """End-to-end scoring test"""
```

### Performance Tests
```python
# test_performance.py
def test_training_speed():
    """Ensure training completes in <10 seconds"""
    
def test_scoring_throughput():
    """Ensure >100 contracts/second"""
```

---

## 📊 Monitoring & Logging

### Log Levels
```python
INFO: Normal operation
WARNING: Overfitting, missing data, etc.
ERROR: Failed operations
DEBUG: Detailed execution flow
```

### Key Metrics to Monitor
- Training F1 score
- Test F1 score
- Train-test gap
- OOB score
- Feature extraction success rate
- Model loading time
- Prediction latency

---

## 🔄 Version Management

### Model Versioning
```
Format: model_vYYYYMMDD_HHMMSS
Example: model_v20260207_183120

Tracks:
├─ Training timestamp
├─ Feature schema version
├─ Hyperparameters
└─ Performance metrics
```

### Backward Compatibility
- Feature schema must match
- Scaler must be compatible
- Model format (joblib) stable

---

## 🚀 Deployment Considerations

### Production Checklist
- [ ] Use fixed model version (don't auto-update)
- [ ] Monitor prediction latency
- [ ] Set up alerting for low confidence predictions
- [ ] Log all predictions for audit
- [ ] Regular retraining schedule
- [ ] A/B testing for model updates
- [ ] Rollback plan for bad models

### Scaling Strategies
1. **Horizontal**: Multiple instances behind load balancer
2. **Vertical**: More CPU/RAM for faster processing
3. **Caching**: Redis for feature/prediction cache
4. **Async**: Queue-based processing for batch jobs

---

**Last Updated:** 2026-02-08  
**Version:** 3.3  
**Maintainer:** ML Unified System Team

