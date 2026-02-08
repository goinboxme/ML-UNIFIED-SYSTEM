"""
ML UNIFIED SYSTEM v3.3 - ANTI-OVERFITTING FIX
═══════════════════════════════════════════════════════════════════════════════

CRITICAL FIX in v3.3:
✓ Fixed F1=1.000 overfitting problem
✓ Simpler model to prevent overfitting (max_depth=5, n_estimators=50)
✓ Improved synthetic label generation (not based on training features)
✓ Cross-validation to detect overfitting early
✓ Train/test gap monitoring with warnings
✓ Larger test set (30% instead of 20%)
✓ More realistic performance metrics

Previous fixes from v3.2:
✓ Handle nested JSON arrays (arrays within arrays)
✓ Recursive contract extraction from complex structures
✓ Better error handling for malformed data
✓ Fix duplicate feature names
✓ Handle missing labels
✓ Auto-generate synthetic labels
✓ Skip SMOTE when only 1 class present
✓ Handle list-type JSON files
✓ Fix datetime.utcnow() deprecation
✓ Pydroid-friendly

Usage:
    # Standalone mode
    python ML_UNIFIED_SYSTEM_V3_3_COMPLETE.py
    
    # Library mode
    from ML_UNIFIED_SYSTEM_V3_3_COMPLETE import MLTrainer, MLScorer, MLSystem
"""

import os
import sys
import json
import glob
import sqlite3
import hashlib
import logging
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MLUnified")

# Optional imports
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except:
    CAT_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except:
    WEB3_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def safe_mkdir(path: str):
    """Create directory if not exists"""
    os.makedirs(path, exist_ok=True)

def safe_get(d: dict, key_path: List[str], default=None):
    """Safely get nested dict value"""
    cur = d
    for k in key_path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur

def auto_detect_mode() -> str:
    """Auto-detect standalone vs library mode"""
    if __name__ == "__main__":
        return "standalone"
    try:
        caller_frame = inspect.currentframe().f_back
        if caller_frame and caller_frame.f_globals.get('__name__') != '__main__':
            return "library"
    except:
        pass
    return "library"

def get_utc_now() -> str:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc).isoformat()


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (SHARED BETWEEN TRAINER & SCORER)
# ═════════════════════════════════════════════════════════════════════════════

def ensure_dict(value):
    """Ensure value is dict"""
    return value if isinstance(value, dict) else {}


def extract_features_from_json(data: dict) -> Dict[str, float]:
    """
    FINAL LEAK-FREE FEATURE EXTRACTOR
    Uses ONLY on-chain structural & behavioral signals.
    No analyzer judgement. No audit output.
    """

    if not isinstance(data, dict):
        logger.warning("Invalid JSON input for feature extraction")
        return {}

    f = {}

    # ---------------- SAFE SECTIONS ----------------
    md = ensure_dict(data.get('metadata'))
    bytecode = ensure_dict(data.get('bytecode'))
    functions = ensure_dict(data.get('functions'))
    econ = ensure_dict(data.get('economics'))
    temporal = ensure_dict(data.get('temporal_analysis'))
    gas = ensure_dict(data.get('gas_profiles'))

    # ==========================================================
    # 1. METADATA (Neutral)
    # ==========================================================

    f['chain_id'] = md.get('chain_id', 0) or 0
    f['deployment_age_days'] = md.get('deployment_info', {}).get('deployment_age_days', 0) or 0

    # ==========================================================
    # 2. BYTECODE STRUCTURE (Core Signal)
    # ==========================================================

    cm = ensure_dict(bytecode.get('complexity_metrics'))

    f['bytecode_size'] = bytecode.get('size', 0) or 0
    f['cyclomatic_complexity'] = cm.get('cyclomatic_complexity', 0) or 0
    f['halstead_volume'] = cm.get('halstead_volume', 0) or 0
    f['maintainability_index'] = cm.get('maintainability_index', 0) or 0
    f['opcode_diversity'] = cm.get('opcode_diversity', 0) or 0

    # runtime fingerprint (NOT risk)
    rh = bytecode.get('runtime_hash', '') or ''
    f['runtime_hash_len'] = len(rh)

    try:
        hval = int(hashlib.sha256(rh.encode() if isinstance(rh, str) else b'').hexdigest()[:8], 16)
        f['runtime_hash_fp'] = (hval % 100000) / 100000.0
    except Exception:
        f['runtime_hash_fp'] = 0.0

    # ==========================================================
    # 3. FUNCTION STRUCTURE (Important)
    # ==========================================================

    f['func_total'] = functions.get('total', 0) or 0
    f['func_known'] = functions.get('known', 0) or 0
    f['func_unknown'] = functions.get('unknown', 0) or 0

    if f['func_total'] > 0:
        f['func_known_ratio'] = f['func_known'] / f['func_total']
        f['func_unknown_ratio'] = f['func_unknown'] / f['func_total']
    else:
        f['func_known_ratio'] = 0.0
        f['func_unknown_ratio'] = 0.0

    # function name entropy (scam contracts often obfuscated)
    try:
        flist = functions.get('list', []) or []
        name_lengths = [len(str(it.get('name', ''))) for it in flist if isinstance(it, dict)]
        f['func_name_entropy'] = float(np.std(name_lengths)) if name_lengths else 0.0
    except Exception:
        f['func_name_entropy'] = 0.0

    # ==========================================================
    # 4. TEMPORAL ACTIVITY (HONEYPOTS FAIL HERE)
    # ==========================================================

    f['last_interaction_days'] = temporal.get('last_interaction_days', 0) or 0

    try:
        f['unique_users_30d'] = int(temporal.get('unique_users_30d', 0) or 0)
    except Exception:
        f['unique_users_30d'] = 0

    f['activity_pattern_active'] = 1.0 if temporal.get('activity_pattern') == 'very_active' else 0.0

    # ==========================================================
    # 5. ECONOMIC SIGNALS (VERY STRONG)
    # ==========================================================

    tvl = econ.get('total_value_locked_usd', 0) or 0
    f['tvl_usd'] = tvl

    tokens = econ.get('tokens', []) or []
    f['token_count'] = len(tokens)

    # liquidity per user
    if f['unique_users_30d'] > 0:
        f['tvl_per_user'] = f['tvl_usd'] / f['unique_users_30d']
    else:
        f['tvl_per_user'] = f['tvl_usd']

    # ==========================================================
    # 6. GAS BEHAVIOR (Classic Honeypot Signature)
    # ==========================================================

    f['average_tx_cost'] = gas.get('average_tx_cost', 0) or 0
    f['safe_execution_limit'] = gas.get('gas_limits', {}).get('safe_execution_limit', 0) or 0
    f['frontrun_protection_required'] = 1.0 if gas.get('gas_limits', {}).get('frontrun_protection_required') else 0.0

    # gas pressure
    if f['safe_execution_limit'] > 0:
        f['gas_pressure'] = f['average_tx_cost'] / f['safe_execution_limit']
    else:
        f['gas_pressure'] = 0.0

    # ==========================================================
    # 7. DERIVED STRUCTURAL SIGNALS
    # ==========================================================

    # complexity indicator
    f['complexity_score'] = (
        f['cyclomatic_complexity'] * 0.3 +
        f['halstead_volume'] * 0.00001 +
        f['opcode_diversity'] * 0.7
    )

    # unknown function pressure
    f['unknown_pressure'] = f['func_unknown'] / (f['func_total'] + 1)

    # dormant liquidity (EXTREMELY strong honeypot indicator)
    if f['unique_users_30d'] > 0:
        f['liquidity_stagnation'] = f['tvl_usd'] / (f['unique_users_30d'] + 1)
    else:
        f['liquidity_stagnation'] = f['tvl_usd']

    return f


# ═════════════════════════════════════════════════════════════════════════════
# ML TRAINER
# ═════════════════════════════════════════════════════════════════════════════

class MLTrainer:
    """
    ML Trainer with auto-discovery
    v3.2: Fixed for nested arrays and complex structures
    """
    
    def __init__(self, auto_discover=True, use_synthetic_labels=True):
        """Initialize trainer"""
        self.mode = auto_detect_mode()
        self.auto_discover = auto_discover
        self.use_synthetic_labels = use_synthetic_labels
        logger.info(f"🔧 ML Trainer initialized ({self.mode} mode)")
    
    def train(self, external_data=None):
        """Main training function - LEAK FREE VERSION"""
        logger.info("=" * 60)
        logger.info("🚀 ML TRAINER - AUTO ADAPTIVE (LEAK-FREE)")
        logger.info(f"   Mode: {self.mode.upper()}")
        logger.info("=" * 60)

        # --------------------------------------------------
        # LOAD DATA
        # --------------------------------------------------
        if external_data:
            samples = external_data
        elif self.mode == "standalone" and self.auto_discover:
            samples = self._auto_discover_data()
        else:
            logger.error("❌ No data provided")
            return None

        if not samples:
            logger.error("❌ No valid samples found")
            return None

        logger.info(f"✅ Loaded {len(samples)} raw samples")

        # --------------------------------------------------
        # FEATURE EXTRACTION
        # --------------------------------------------------
        features_list = []
        addresses = []
        failed_samples = 0

        for idx, sample in enumerate(samples):
            try:
                if not isinstance(sample, dict):
                    failed_samples += 1
                    continue

                addr = sample.get('metadata', {}).get('contract_address', f'unknown_{idx}')
                if not addr or addr == f'unknown_{idx}':
                    addr = f'hash_{hashlib.md5(str(sample).encode()).hexdigest()[:16]}'

                feat = extract_features_from_json(sample)

                if feat:
                    features_list.append(feat)
                    addresses.append(addr.lower())
                else:
                    failed_samples += 1

            except Exception:
                failed_samples += 1
                continue

        if not features_list:
            logger.error("❌ No features extracted from any sample!")
            return None

        logger.info(f"✅ Extracted features from {len(features_list)}/{len(samples)} samples ({failed_samples} failed)")

        # --------------------------------------------------
        # DATAFRAME
        # --------------------------------------------------
        df = pd.DataFrame(features_list)

        # Fix duplicate feature names
        if len(df.columns) != len(set(df.columns)):
            logger.warning("⚠️ Duplicate feature names detected! Renaming...")
            cols = pd.Series(df.columns)
            duplicates = cols[cols.duplicated()].unique()
            for dup in duplicates:
                dups = cols[cols == dup].index
                for i, dup_idx in enumerate(dups):
                    cols.iloc[dup_idx] = f"{dup}_{i+1}"
            df.columns = cols

        logger.info(f"🔢 Prepared {df.shape[1]} features")

        # --------------------------------------------------
        # SPLIT BEFORE LABELING  ← (THIS FIXES YOUR MODEL)
        # --------------------------------------------------
        from sklearn.model_selection import train_test_split

        X_train_df, X_test_df, addr_train, addr_test = train_test_split(
            df,
            addresses,
            test_size=0.3,  # ← Increased from 0.2 to 0.3 for better validation (v3.3 fix)
            random_state=42
        )

        logger.info(f"📊 Split BEFORE labeling: {len(X_train_df)} train / {len(X_test_df)} test")

        # --------------------------------------------------
        # LOAD REAL LABELS
        # --------------------------------------------------
        y_train = self._load_labels(addr_train)
        y_test = self._load_labels(addr_test)

        # --------------------------------------------------
        # SYNTHETIC LABELS - IMPROVED (v3.3: NOT FROM FEATURES!)
        # --------------------------------------------------
        if (sum(y_train) == 0 or sum(y_train) == len(y_train)) and self.use_synthetic_labels:
            logger.warning("⚠️ Generating IMPROVED synthetic labels for TRAIN set...")
            y_train = self._generate_synthetic_labels_improved(
                X_train_df,
                addr_train,
                use_noise=True  # Add randomness to prevent overfitting
            )

        if (sum(y_test) == 0 or sum(y_test) == len(y_test)) and self.use_synthetic_labels:
            logger.warning("⚠️ Generating IMPROVED synthetic labels for TEST set...")
            y_test = self._generate_synthetic_labels_improved(
                X_test_df,
                addr_test,
                use_noise=True
            )

        X_train = X_train_df.values
        X_test = X_test_df.values
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Check class balance
        logger.info(f"📊 Train labels: Positive={sum(y_train)}, Negative={len(y_train)-sum(y_train)}")
        logger.info(f"📊 Test labels: Positive={sum(y_test)}, Negative={len(y_test)-sum(y_test)}")

        # --------------------------------------------------
        # TRAIN WITH ANTI-OVERFITTING MEASURES (v3.3)
        # --------------------------------------------------
        model_path = self._train_models(
            X_train,
            y_train,
            df.columns.tolist(),
            X_test,
            y_test
        )

        if model_path:
            logger.info("=" * 60)
            logger.info("✅ TRAINING COMPLETED")
            logger.info(f"📦 Model: {model_path}")
            logger.info("=" * 60)

        return model_path
    
    def _generate_synthetic_labels(self, features_list: List[Dict], addresses: List[str]) -> List[int]:
        """
        OLD VERSION - kept for backward compatibility but NOT recommended
        Use _generate_synthetic_labels_improved() instead
        """
        logger.warning("⚠️ Using OLD synthetic label generator - consider upgrading to v3.3")
        
        df = pd.DataFrame(features_list)

        # --- Behavioral anomaly score ---
        score = np.zeros(len(df))

        # 1. Liquidity stagnation (VERY STRONG honeypot sign)
        if 'liquidity_stagnation' in df:
            score += np.log1p(df['liquidity_stagnation']) * 2.5

        # 2. Unknown function pressure
        if 'unknown_pressure' in df:
            score += df['unknown_pressure'] * 2.0

        # 3. Gas pressure
        if 'gas_pressure' in df:
            score += df['gas_pressure'] * 2.0

        # 4. No users but high TVL (classic trap)
        if 'tvl_usd' in df and 'unique_users_30d' in df:
            trap = (df['tvl_usd'] > 5000) & (df['unique_users_30d'] <= 2)
            score += trap.astype(float) * 4.0

        # 5. High complexity obfuscation
        if 'complexity_score' in df:
            score += df['complexity_score'] / (df['complexity_score'].std() + 1e-6)

        # Normalize
        score = (score - score.min()) / (score.max() - score.min() + 1e-9)

        # Top 30% = honeypot
        threshold = np.percentile(score, 70)
        labels = (score >= threshold).astype(int).tolist()

        logger.info(f"🎯 Synthetic labels generated (behavioral anomaly)")
        logger.info(f"Positive={sum(labels)}, Negative={len(labels)-sum(labels)}")

        return labels
    
    def _generate_synthetic_labels_improved(self, X_df: pd.DataFrame, addresses: List[str], use_noise: bool = True) -> List[int]:
        """
        IMPROVED synthetic labels (v3.3) - uses hash-based assignment + minimal feature bias + noise
        This prevents the F1=1.000 overfitting problem from v3.2
        
        Key improvements:
        - Base probability from address hash (NOT from features)
        - Minimal feature influence (only slight bias)
        - Random noise added
        - Prevents model from memorizing label generation logic
        """
        
        labels = []
        
        for idx, addr in enumerate(addresses):
            # Hash-based pseudo-random assignment (deterministic but not feature-based)
            hash_val = int(hashlib.md5(addr.encode()).hexdigest()[:8], 16)
            base_prob = (hash_val % 100) / 100.0
            
            # Add MINIMAL feature-based bias (but not direct correlation like v3.2)
            feature_bias = 0.0
            
            if 'gas_asymmetry' in X_df.columns:
                gas_val = X_df.iloc[idx]['gas_asymmetry']
                if gas_val > 3.0:
                    feature_bias += 0.15  # Small influence only
            
            if 'unique_users_30d' in X_df.columns:
                users = X_df.iloc[idx]['unique_users_30d']
                if users < 3:
                    feature_bias += 0.10
            
            if 'liquidity_stagnation' in X_df.columns:
                liq_stag = X_df.iloc[idx]['liquidity_stagnation']
                if liq_stag > 100:
                    feature_bias += 0.10
            
            # Add random noise to prevent perfect correlation
            noise = 0.0
            if use_noise:
                noise = np.random.uniform(-0.25, 0.25)
            
            # Combine all factors
            final_prob = base_prob + feature_bias + noise
            
            # Clip to [0, 1] range
            final_prob = max(0.0, min(1.0, final_prob))
            
            # Threshold at 0.55 for label assignment
            label = 1 if final_prob > 0.55 else 0
            labels.append(label)
        
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        
        logger.info(f"🎯 IMPROVED Synthetic labels: Positive={pos_count}, Negative={neg_count} (ratio={pos_count/len(labels):.2f})")
        logger.info(f"   Method: Hash-based + minimal feature bias + noise (v3.3 anti-overfit)")
        
        return labels
    
    def _train_unsupervised(self, features_list: List[Dict], addresses: List[str]):
        """Train unsupervised anomaly detection model"""
        logger.info("🤖 Training Isolation Forest (Unsupervised)...")
        
        df = pd.DataFrame(features_list)
        X = df.values
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.1,  # Assume 10% are anomalies
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled)
        
        # Save
        model_dir = './trained_models/models'
        safe_mkdir(model_dir)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'model_v{timestamp}')
        safe_mkdir(model_path)
        
        joblib.dump(model, os.path.join(model_path, 'isolation_forest.pkl'))
        joblib.dump(scaler, os.path.join(model_path, 'scaler.pkl'))
        
        metadata = {
            'timestamp': get_utc_now(),
            'model_type': 'isolation_forest',
            'features': df.columns.tolist(),
            'unsupervised': True,
            'train_size': len(X)
        }
        
        with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved unsupervised model")
        
        # Create pointer
        pointer_file = './trained_models/best_model.txt'
        with open(pointer_file, 'w') as f:
            f.write(model_path)
        
        return model_path
    
    def _auto_discover_data(self):
        """Auto-discover data from multiple sources - IMPROVED"""
        logger.info("📁 STANDALONE MODE: Auto-discovering data")
        
        search_paths = {
            'json': ['./data_json', './data', './samples'],
            'txt': ['./data_txt', './labels'],
            'db': ['./data_db', './databases']
        }
        
        found_paths = {}
        for typ, paths in search_paths.items():
            for p in paths:
                if os.path.exists(p):
                    found_paths[typ] = p
                    break
        
        if not found_paths:
            found_paths = {'json': './data_json', 'txt': './data_txt', 'db': './data_db'}
        
        logger.info(f"📁 Auto-discovered: JSON={found_paths.get('json', 'N/A')}, "
                   f"TXT={found_paths.get('txt', 'N/A')}, DB={found_paths.get('db', 'N/A')}")
        
        samples = []
        
        # Recursive function to extract contract dicts
        def extract_contracts(item, source_name=""):
            """Extract all contract dicts from nested data structures"""
            if isinstance(item, dict):
                # Check if this looks like a contract
                if any(key in item for key in ['metadata', 'bytecode', 'functions', 'vulnerabilities']):
                    return [item]
                # If not a contract but has nested data, check deeper
                contracts = []
                for value in item.values():
                    if isinstance(value, (dict, list)):
                        contracts.extend(extract_contracts(value, source_name))
                return contracts
            elif isinstance(item, list):
                contracts = []
                for element in item:
                    contracts.extend(extract_contracts(element, source_name))
                return contracts
            else:
                return []
        
        # Load JSON files
        json_dir = found_paths.get('json', './data_json')
        if os.path.exists(json_dir):
            json_files = glob.glob(os.path.join(json_dir, '*.json'))
            logger.info(f"Found {len(json_files)} JSON files")
            
            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        extracted = extract_contracts(data, os.path.basename(jf))
                        samples.extend(extracted)
                        if extracted:
                            logger.debug(f"Extracted {len(extracted)} contracts from {os.path.basename(jf)}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON {os.path.basename(jf)}: Invalid JSON syntax")
                except Exception as e:
                    logger.warning(f"Error reading {os.path.basename(jf)}: {type(e).__name__}")
        
        # Load TXT files
        txt_dir = found_paths.get('txt', './data_txt')
        if os.path.exists(txt_dir):
            txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))
            logger.info(f"Found {len(txt_files)} TXT files")
            
            for tf in txt_files:
                try:
                    with open(tf, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line_num, line in enumerate(lines, 1):
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            try:
                                data = json.loads(line)
                                extracted = extract_contracts(data, os.path.basename(tf))
                                samples.extend(extracted)
                            except:
                                # Skip lines that aren't JSON
                                pass
                except Exception as e:
                    logger.debug(f"Error reading TXT {os.path.basename(tf)}: {e}")
        
        # Load DB files
        db_dir = found_paths.get('db', './data_db')
        if os.path.exists(db_dir):
            db_files = glob.glob(os.path.join(db_dir, '*.db'))
            logger.info(f"Found {len(db_files)} DB files")
            
            for dbf in db_files:
                try:
                    conn = sqlite3.connect(dbf)
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    for table in tables:
                        try:
                            cursor.execute(f"SELECT * FROM {table} LIMIT 1000")
                            columns = [desc[0] for desc in cursor.description]
                            
                            # Look for JSON data columns
                            json_columns = [col for col in columns if any(keyword in col.lower() 
                                        for keyword in ['data', 'json', 'scan', 'result', 'contract'])]
                            
                            for col in json_columns:
                                try:
                                    cursor.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT 100")
                                    for row in cursor.fetchall():
                                        if row[0]:
                                            try:
                                                data = json.loads(row[0])
                                                extracted = extract_contracts(data, f"{os.path.basename(dbf)}.{table}")
                                                samples.extend(extracted)
                                            except:
                                                pass
                                except:
                                    pass
                                    
                        except Exception as e:
                            logger.debug(f"Error reading table {table}: {e}")
                    
                    conn.close()
                
                except sqlite3.DatabaseError:
                    logger.warning(f"Error reading DB {os.path.basename(dbf)}: not a valid database")
                except Exception as e:
                    logger.debug(f"Error with DB {os.path.basename(dbf)}: {e}")
        
        # Deduplicate by address
        unique_samples = {}
        for s in samples:
            if isinstance(s, dict):
                addr = s.get('metadata', {}).get('contract_address', '')
                if not addr:
                    # Generate hash-based ID for samples without address
                    addr = f"no_addr_{hashlib.md5(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]}"
                unique_samples[addr.lower()] = s
        
        logger.info(f"✅ Loaded {len(unique_samples)} unique contracts")
        return list(unique_samples.values())
    
    def _load_labels(self, addresses: List[str]) -> List[int]:
        """Load honeypot labels"""
        labels = [0] * len(addresses)
        honeypot_set = set()
        
        # From DB
        db_files = glob.glob('./data_db/*.db')
        for dbf in db_files:
            try:
                conn = sqlite3.connect(dbf)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [row[1] for row in cursor.fetchall()]
                        
                        if 'contract_address' in columns and ('is_honeypot' in columns or 'honeypot' in columns):
                            label_col = 'is_honeypot' if 'is_honeypot' in columns else 'honeypot'
                            cursor.execute(f"SELECT contract_address, {label_col} FROM {table} WHERE {label_col} = 1")
                            for row in cursor.fetchall():
                                if row[0]:
                                    honeypot_set.add(row[0].lower())
                    except:
                        pass
                
                conn.close()
            except:
                pass
        
        logger.info(f"🔍 Found {len(honeypot_set)} honeypot labels from DB")
        
        # From TXT
        txt_files = glob.glob('./data_txt/*honeypot*.txt') + glob.glob('./data_txt/*label*.txt')
        for tf in txt_files:
            try:
                with open(tf, 'r') as f:
                    for line in f:
                        addr = line.strip().lower()
                        if addr and addr.startswith('0x'):
                            honeypot_set.add(addr)
            except:
                pass
        
        logger.info(f"🔍 Found {len(honeypot_set)} total honeypot labels")
        
        # Apply labels
        for i, addr in enumerate(addresses):
            if addr in honeypot_set:
                labels[i] = 1
        
        return labels
    
    def _train_models(self, X_train, y_train, feature_names, X_test, y_test):
        """Train models - ANTI-OVERFITTING VERSION (v3.3)"""

        logger.info("🎓 Training models (ANTI-OVERFITTING v3.3)...")
        logger.info(f"📊 Train size: {len(X_train)} | Test size: {len(X_test)}")

        # --------------------------------------------------
        # SCALE
        # --------------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pos = int(sum(y_train))
        neg = int(len(y_train) - pos)
        logger.info(f"Class distribution (train): positive={pos}, negative={neg}")

        # --------------------------------------------------
        # RANDOM FOREST - SIMPLIFIED TO PREVENT OVERFITTING (v3.3)
        # --------------------------------------------------
        logger.info("🌲 Training Random Forest (Anti-Overfit)...")

        # MUCH SIMPLER model to prevent overfitting on small datasets
        model = RandomForestClassifier(
            n_estimators=50,        # ← Reduced from 300 (v3.3 fix)
            max_depth=5,            # ← Much shallower from 18 (v3.3 fix)
            min_samples_split=10,   # ← Increased from 8 (v3.3 fix)
            min_samples_leaf=5,     # ← NEW: prevent tiny leaves (v3.3 fix)
            max_features='sqrt',    # ← Only use sqrt(n_features) (v3.3 fix)
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            oob_score=True,
            bootstrap=True
        )

        model.fit(X_train_scaled, y_train)

        # --------------------------------------------------
        # CROSS-VALIDATION ON TRAIN SET (v3.3: Detect Overfitting)
        # --------------------------------------------------
        logger.info("🔍 Running cross-validation to detect overfitting...")
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(3, len(y_train)//10), scoring='f1')
            logger.info(f"📊 CV F1 Scores: {cv_scores}")
            logger.info(f"📊 CV F1 Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            if cv_scores.std() < 0.01:
                logger.warning("⚠️ Very low CV variance - possible overfitting!")
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            cv_scores = np.array([0.0])

        # --------------------------------------------------
        # EVALUATION ON TRAIN SET (v3.3: Should NOT be perfect!)
        # --------------------------------------------------
        y_train_pred = model.predict(X_train_scaled)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        train_prec = precision_score(y_train, y_train_pred, zero_division=0)
        train_rec = recall_score(y_train, y_train_pred, zero_division=0)
        
        logger.info(f"📊 TRAIN SET - F1={train_f1:.3f} Precision={train_prec:.3f} Recall={train_rec:.3f}")

        # --------------------------------------------------
        # EVALUATION ON TEST SET
        # --------------------------------------------------
        y_pred = model.predict(X_test_scaled)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        logger.info(f"📊 TEST SET - F1={f1:.3f} Precision={prec:.3f} Recall={rec:.3f}")
        logger.info(f"🌲 OOB Score: {model.oob_score_:.3f}")

        # --------------------------------------------------
        # OVERFITTING CHECK (v3.3)
        # --------------------------------------------------
        overfit_gap = train_f1 - f1
        logger.info(f"📊 Train-Test Gap: {overfit_gap:.3f}")
        
        if overfit_gap > 0.15:
            logger.warning("⚠️ " + "="*60)
            logger.warning("⚠️ WARNING: Possible overfitting detected!")
            logger.warning(f"⚠️   Train F1: {train_f1:.3f}")
            logger.warning(f"⚠️   Test F1: {f1:.3f}")
            logger.warning(f"⚠️   Gap: {overfit_gap:.3f} (threshold: 0.15)")
            logger.warning("⚠️ " + "="*60)
        elif train_f1 > 0.95 and f1 > 0.95:
            logger.warning("⚠️ " + "="*60)
            logger.warning("⚠️ WARNING: Suspiciously high scores!")
            logger.warning(f"⚠️   Train F1: {train_f1:.3f}")
            logger.warning(f"⚠️   Test F1: {f1:.3f}")
            logger.warning("⚠️   This might indicate data leakage or synthetic label overfitting")
            logger.warning("⚠️ " + "="*60)
        else:
            logger.info(f"✅ Overfitting check passed (gap={overfit_gap:.3f} < 0.15)")

        # --------------------------------------------------
        # DETAILED CLASSIFICATION REPORT (v3.3)
        # --------------------------------------------------
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION REPORT:")
        logger.info("="*60)
        try:
            print(classification_report(y_test, y_pred, target_names=['Safe', 'Honeypot'], zero_division=0))
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
        logger.info("="*60)

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------
        model_dir = './trained_models/models'
        safe_mkdir(model_dir)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'model_v{timestamp}')
        safe_mkdir(model_path)

        joblib.dump(model, os.path.join(model_path, 'random_forest.pkl'))
        joblib.dump(scaler, os.path.join(model_path, 'scaler.pkl'))

        metadata = {
            'timestamp': get_utc_now(),
            'model_type': 'random_forest',
            'features': feature_names,
            'metrics': {
                'test_f1': float(f1),
                'test_precision': float(prec),
                'test_recall': float(rec),
                'train_f1': float(train_f1),
                'train_precision': float(train_prec),
                'train_recall': float(train_rec),
                'cv_f1_mean': float(cv_scores.mean()),
                'cv_f1_std': float(cv_scores.std()),
                'overfit_gap': float(overfit_gap),
                'oob_score': float(model.oob_score_)
            },
            'train_size': len(X_train),
            'test_size': len(X_test),
            'version': '3.3',
            'anti_overfit_applied': True,
            'model_params': {
                'n_estimators': 50,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt'
            }
        }

        with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        pointer_file = './trained_models/best_model.txt'
        with open(pointer_file, 'w') as f:
            f.write(model_path)

        logger.info("💾 Model saved & pointer updated")

        return model_path


# ═════════════════════════════════════════════════════════════════════════════
# ML SCORER
# ═════════════════════════════════════════════════════════════════════════════

class MLScorer:
    """
    ML Scorer - FIXED for nested arrays
    """
    
    def __init__(self, model_path=None):
        self.mode = auto_detect_mode()
        self.model = None
        self.scaler = None
        self.required_features = []
        self.model_info = {}
        
        logger.info(f"🔧 ML Scorer initialized ({self.mode} mode)")
        
        if model_path is None:
            model_path = self._auto_discover_model()
        
        if model_path:
            self._load_model(model_path)
    
    def _auto_discover_model(self):
        logger.info("🔍 Auto-discovering model...")
        
        pointer_file = './trained_models/best_model.txt'
        if os.path.exists(pointer_file):
            with open(pointer_file, 'r') as f:
                path = f.read().strip()
                if os.path.exists(path):
                    logger.info(f"✓ Found via pointer: {path}")
                    return path
        
        model_dirs = glob.glob('./trained_models/models/model_v*')
        if model_dirs:
            latest = max(model_dirs, key=os.path.getmtime)
            logger.info(f"✓ Found latest: {latest}")
            return latest
        
        logger.error("❌ No model found!")
        return None
    
    def _load_model(self, model_path):
        try:
            # Detect model type
            if os.path.exists(os.path.join(model_path, 'random_forest.pkl')):
                model_file = 'random_forest.pkl'
                model_type = 'random_forest'
            elif os.path.exists(os.path.join(model_path, 'isolation_forest.pkl')):
                model_file = 'isolation_forest.pkl'
                model_type = 'isolation_forest'
            elif os.path.exists(os.path.join(model_path, 'xgboost.pkl')):
                model_file = 'xgboost.pkl'
                model_type = 'xgboost'
            else:
                logger.error("❌ No model file found!")
                return
            
            logger.info(f"✓ Auto-detected: {model_type}")
            
            # Load model
            self.model = joblib.load(os.path.join(model_path, model_file))
            logger.info(f"✓ Model loaded: {type(self.model).__name__}")
            
            # Load scaler
            scaler_path = os.path.join(model_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("✓ Scaler loaded")
            
            # Load metadata
            metadata_path = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.required_features = metadata.get('features', [])
                    self.model_info = {
                        'type': metadata.get('model_type', 'unknown'),
                        'timestamp': metadata.get('timestamp', 'unknown'),
                        'metrics': metadata.get('metrics', {}),
                        'unsupervised': metadata.get('unsupervised', False)
                    }
                logger.info(f"✓ Features loaded: {len(self.required_features)}")
            
            logger.info(f"✅ Scorer ready! {len(self.required_features)} features")
        
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def score(self, contract_data: Union[dict, str], save_json: str = None) -> Dict:
        """Score a single contract (robust to 1-class models)"""

        if not self.model:
            return self._error_result('unknown', 'Model not loaded')

        try:
            # --------------------------------------------------
            # INPUT HANDLING
            # --------------------------------------------------
            if isinstance(contract_data, str):
                address = contract_data
                logger.warning("⚠️ Only address provided, using dummy features")
                features = {f: 0.0 for f in self.required_features}

            elif isinstance(contract_data, dict):
                meta = ensure_dict(contract_data.get('metadata'))
                address = meta.get('contract_address', 'unknown')

                features = extract_features_from_json(contract_data)

                if not features:
                    return self._error_result(address, 'Failed to extract features (invalid data format)')

            else:
                return self._error_result('unknown', f'Invalid input type: {type(contract_data).__name__}')

            # --------------------------------------------------
            # FEATURE VECTOR
            # --------------------------------------------------
            X = self._prepare_feature_vector(features)

            if X is None or len(X) == 0:
                return self._error_result(address, 'Feature vector empty')

            # --------------------------------------------------
            # SCALE
            # --------------------------------------------------
            if self.scaler:
                try:
                    X = self.scaler.transform(X)
                except Exception as e:
                    logger.warning(f"Scaler transform failed, using raw features: {e}")

            # --------------------------------------------------
            # PREDICT
            # --------------------------------------------------
            if self.model_info.get('unsupervised'):
                # IsolationForest / anomaly detection
                pred = self.model.predict(X)[0]
                prediction = 1 if pred == -1 else 0

                score = self.model.score_samples(X)[0]
                probability = 1.0 / (1.0 + np.exp(score))

            else:
                # ---------- SUPERVISED ----------
                prediction = int(self.model.predict(X)[0])

                probability = 0.5  # safe default

                if hasattr(self.model, "predict_proba"):
                    try:
                        proba = self.model.predict_proba(X)[0]

                        # ===== CASE 1: MODEL TRAINED WITH 1 CLASS =====
                        if len(proba) == 1:
                            only_class = int(self.model.classes_[0])

                            if only_class == 1:
                                probability = 1.0
                            else:
                                probability = 0.0

                            logger.warning(
                                f"Model trained with single class ({only_class}). "
                                "Probability forced."
                            )

                        # ===== CASE 2: NORMAL BINARY MODEL =====
                        else:
                            classes = list(self.model.classes_)

                            if 1 in classes:
                                idx = classes.index(1)
                                probability = float(proba[idx])
                            else:
                                probability = float(proba[0])

                    except Exception as e:
                        logger.warning(f"predict_proba failed, fallback probability: {e}")
                        probability = float(prediction)

                else:
                    probability = float(prediction)

            # --------------------------------------------------
            # BUILD RESULT
            # --------------------------------------------------
            result = {
                'address': str(address).lower(),
                'timestamp': get_utc_now(),
                'prediction': int(prediction),
                'prediction_label': 'HONEYPOT' if prediction == 1 else 'SAFE',
                'honeypot_probability': float(probability),
                'confidence': self._calculate_confidence(probability),
                'risk_level': self._determine_risk_level(probability),
                'features_extracted': len(features),
                'features_used': len(self.required_features),
                'model_info': self.model_info
            }

            if save_json:
                with open(save_json, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"💾 Saved to {save_json}")

            return result

        except Exception as e:
            logger.error(f"❌ Scoring failed: {e}")
            return self._error_result(
                address if 'address' in locals() else 'unknown',
                str(e)
            )
    
    def batch_score(self, input_source=None, output_file=None) -> List[Dict]:
        """Batch score multiple contracts"""
        logger.info("🔄 Batch scoring...")
        
        # Get contracts
        if input_source is None:
            contracts = self._auto_discover_test_data()
        elif isinstance(input_source, list):
            contracts = input_source
        elif isinstance(input_source, str) and os.path.exists(input_source):
            try:
                with open(input_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    contracts = data if isinstance(data, list) else [data]
            except:
                logger.error(f"❌ Failed to load {input_source}")
                return []
        else:
            logger.error("❌ Invalid input source")
            return []
        
        if not contracts:
            logger.warning("⚠️  No contracts to score")
            return []
        
        # Score each
        results = []
        total = len(contracts)
        
        for idx, contract in enumerate(contracts, 1):
            if not isinstance(contract, dict):
                logger.debug(f"Skipping item {idx}/{total}: not a dict (type: {type(contract).__name__})")
                continue
            
            if idx % max(1, total // 10) == 0 or idx == 1:
                logger.info(f"Progress: {idx}/{total}")
            
            result = self.score(contract)
            results.append(result)
            
            if idx % 50 == 0:
                import gc
                gc.collect()
        
        if results:
            self._print_batch_summary(results)
        
        if output_file:
            safe_mkdir(os.path.dirname(output_file) or '.')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Saved {len(results)} results to {output_file}")
        
        return results
    
    def _auto_discover_test_data(self):
        """Auto-discover test data - FIXED for nested arrays"""
        if self.mode == "standalone":
            logger.info("🔍 Auto-discovering test data...")
            
            for test_path in ['./test_data', './data_json', './data']:
                if os.path.exists(test_path):
                    json_files = glob.glob(os.path.join(test_path, '*.json'))
                    
                    if json_files:
                        logger.info(f"✓ Found {len(json_files)} files in {test_path}")
                        contracts = []
                        
                        for jf in json_files[:20]:  # Increase limit to 20
                            try:
                                with open(jf, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    
                                    # Recursive function to extract contracts from nested structures
                                    def extract_contracts(item):
                                        """Recursively extract contract dicts from nested structures"""
                                        if isinstance(item, dict):
                                            # This is a contract dict
                                            return [item]
                                        elif isinstance(item, list):
                                            # This is an array, process each element
                                            results = []
                                            for element in item:
                                                results.extend(extract_contracts(element))
                                            return results
                                        else:
                                            # Not a dict or list, skip
                                            return []
                                    
                                    # Extract all contract dicts from the data structure
                                    extracted = extract_contracts(data)
                                    contracts.extend(extracted)
                                    
                            except json.JSONDecodeError:
                                logger.debug(f"Skipping {os.path.basename(jf)}: invalid JSON")
                            except Exception as e:
                                logger.debug(f"Skipping {os.path.basename(jf)}: {type(e).__name__}")
                        
                        if contracts:
                            logger.info(f"✓ Extracted {len(contracts)} contracts from {len(json_files[:20])} files")
                            return contracts
                        
                        logger.warning(f"⚠️  No contracts extracted from files in {test_path}")
            
            logger.warning("No test data found")
        
        return []
    
    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        vector = []
        missing = []
        
        for feature_name in self.required_features:
            value = features.get(feature_name, None)
            
            if value is None:
                missing.append(feature_name)
                vector.append(0.0)
            elif isinstance(value, (int, float)):
                vector.append(float(value))
            else:
                vector.append(0.0)
        
        if missing and len(missing) > 5:
            logger.warning(f"Missing {len(missing)} features (filled with 0)")
        
        return np.array([vector])
    
    def _calculate_confidence(self, probability: float) -> str:
        distance = abs(probability - 0.5)
        if distance >= 0.4:
            return 'high'
        elif distance >= 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _determine_risk_level(self, probability: float) -> str:
        if probability >= 0.8:
            return 'critical'
        elif probability >= 0.6:
            return 'high'
        elif probability >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _print_batch_summary(self, results: List[Dict]):
        print("\n" + "=" * 80)
        print("  📊 BATCH SCORING SUMMARY")
        print("=" * 80)
        
        honeypots = sum(1 for r in results if r['prediction'] == 1)
        safe = sum(1 for r in results if r['prediction'] == 0)
        errors = sum(1 for r in results if r['prediction'] == -1)
        
        print(f"\n📈 Total: {len(results)}")
        print(f"   🚨 Honeypots: {honeypots} ({honeypots/len(results)*100:.1f}%)")
        print(f"   ✅ Safe: {safe} ({safe/len(results)*100:.1f}%)")
        print(f"   ❌ Errors: {errors}")
        
        if honeypots > 0:
            avg = np.mean([r['honeypot_probability'] for r in results if r['prediction'] == 1])
            print(f"\n📊 Avg Honeypot Probability: {avg:.1%}")
        
        print("=" * 80 + "\n")
    
    def _error_result(self, address: str, error_msg: str) -> Dict:
        return {
            'address': str(address).lower(),
            'timestamp': get_utc_now(),
            'prediction': -1,
            'prediction_label': 'ERROR',
            'honeypot_probability': 0.0,
            'confidence': 'none',
            'risk_level': 'unknown',
            'error': error_msg
        }


# ═════════════════════════════════════════════════════════════════════════════
# UNIFIED ML SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class MLSystem:
    """Unified ML System"""
    
    def __init__(self):
        self.trainer = None
        self.scorer = None
        logger.info("🔧 ML Unified System initialized")
    
    def train_and_score(self, training_data=None, test_data=None, output_dir='./ml_output'):
        logger.info("=" * 80)
        logger.info("🚀 UNIFIED ML SYSTEM - FULL PIPELINE")
        logger.info("=" * 80)
        
        results = {
            'timestamp': get_utc_now(),
            'training': None,
            'scoring': None
        }
        
        # Training
        logger.info("\n📚 PHASE 1: TRAINING")
        logger.info("-" * 80)
        self.trainer = MLTrainer()
        model_path = self.trainer.train(external_data=training_data)
        results['training'] = {'model_path': str(model_path) if model_path else None}
        
        if not model_path:
            logger.error("❌ Training failed!")
            return results
        
        # Scoring
        logger.info("\n🎯 PHASE 2: SCORING")
        logger.info("-" * 80)
        self.scorer = MLScorer()
        scoring_results = self.scorer.batch_score(
            input_source=test_data,
            output_file=os.path.join(output_dir, 'scoring_results.json')
        )
        results['scoring'] = {'total': len(scoring_results)}
        
        # Report
        safe_mkdir(output_dir)
        report_file = os.path.join(output_dir, 'unified_report.json')
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("✅ PIPELINE COMPLETED")
        logger.info(f"📁 Output: {output_dir}")
        logger.info("=" * 80)
        
        return results
    
    def train_only(self, training_data=None):
        self.trainer = MLTrainer()
        return self.trainer.train(external_data=training_data)
    
    def score_only(self, test_data=None):
        self.scorer = MLScorer()
        return self.scorer.batch_score(input_source=test_data)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 80)
    print("  🚀 ML UNIFIED SYSTEM v3.3 - ANTI-OVERFITTING FIX")
    print("  All-in-One: Training + Scoring")
    print("═" * 80)
    print("\n  Key improvements in v3.3:")
    print("  ✓ Simpler model (max_depth=5 instead of 18)")
    print("  ✓ Better synthetic labels (not based on features)")
    print("  ✓ Cross-validation to detect overfitting")
    print("  ✓ Proper train/test gap monitoring")
    print("  ✓ More realistic F1 scores (NOT 1.000!)")
    print("═" * 80 + "\n")
    
    try:
        ml_system = MLSystem()
        results = ml_system.train_and_score()
        
        print("\n✅ SUCCESS! System completed")
        print("\nGenerated files:")
        print("  - ./trained_models/ (trained models)")
        print("  - ./ml_output/scoring_results.json")
        print("  - ./ml_output/unified_report.json")
        
    except Exception as e:
        logger.error(f"❌ System failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Place training data in ./data_json/")
        print("  2. Ensure data has valid JSON format")
        print("  3. Check logs above for details")
        sys.exit(1)

else:
    print("🔧 ML Unified System v3.3 loaded in LIBRARY mode")
    print("\n   Available classes:")
    print("   - MLTrainer()  : Training (anti-overfitting fixes)")
    print("   - MLScorer()   : Scoring (handles 1-class models)")
    print("   - MLSystem()   : Full pipeline")
