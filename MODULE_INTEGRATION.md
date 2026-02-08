# 🔗 Module Integration Guide

How to connect external blockchain analyzers to ML Unified System v3.3

---

## 🎯 Overview

ML Unified System is designed to work with **ANY** blockchain analyzer that can produce structured JSON reports. This guide shows you how to integrate various types of analyzers.

```
Your Analyzer → JSON Report → ML System → Risk Prediction
```

---

## 📋 Required JSON Schema

### Minimal Schema

At minimum, your analyzer should produce JSON with these top-level keys:

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

**Note:** Empty objects `{}` are acceptable if data is unavailable. The ML system will handle missing data gracefully.

### Recommended Full Schema

```json
{
  "metadata": {
    "chain_id": 1,
    "contract_address": "0x...",
    "deployment_info": {
      "deployment_age_days": 45,
      "deployer": "0x...",
      "deployment_tx": "0x..."
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
      {
        "name": "transfer",
        "selector": "0xa9059cbb",
        "visibility": "public"
      }
    ]
  },
  
  "temporal_analysis": {
    "last_interaction_days": 30,
    "unique_users_30d": 150,
    "total_transactions": 5000,
    "activity_pattern": "very_active"
  },
  
  "economics": {
    "total_value_locked_usd": 1500000,
    "tokens": [
      {
        "symbol": "USDC",
        "address": "0x...",
        "balance": "500000",
        "value_usd": 500000
      }
    ],
    "token_count": 2
  },
  
  "gas_profiles": {
    "average_tx_cost": 250000,
    "median_tx_cost": 200000,
    "max_tx_cost": 500000,
    "gas_limits": {
      "safe_execution_limit": 3000000,
      "frontrun_protection_required": false
    }
  }
}
```

---

## 🔌 Integration Patterns

### Pattern 1: Direct Integration (Python)

**Use Case:** Your analyzer is written in Python

```python
# your_analyzer.py
import json
from ml_unified_system import MLScorer

def analyze_and_score(contract_address):
    # Step 1: Run your analysis
    report = {
        "metadata": get_metadata(contract_address),
        "bytecode": analyze_bytecode(contract_address),
        "functions": extract_functions(contract_address),
        "temporal_analysis": get_activity(contract_address),
        "economics": get_economics(contract_address),
        "gas_profiles": profile_gas(contract_address)
    }
    
    # Step 2: ML scoring
    scorer = MLScorer()
    prediction = scorer.score_single(report)
    
    return prediction

# Usage
result = analyze_and_score("0x1234...")
print(f"Risk: {result['prediction']} ({result['probability']:.2%})")
```

### Pattern 2: File-Based Integration

**Use Case:** Your analyzer outputs files

```python
# your_analyzer.py
import json

def analyze_contract(contract_address):
    # Your analysis logic
    report = {
        "metadata": {...},
        "bytecode": {...},
        # ... rest of the report
    }
    
    # Save to JSON file
    filename = f"./data_json/{contract_address}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved: {filename}")

# Later, run ML system
# python ML_UNIFIED_SYSTEM_V3_3.py
```

### Pattern 3: Batch Integration

**Use Case:** Analyze multiple contracts at once

```python
# batch_analyzer.py
import json

def batch_analyze(contract_addresses):
    results = []
    
    for address in contract_addresses:
        report = full_analysis(address)
        
        # Save individual file
        with open(f"./data_json/{address}.json", 'w') as f:
            json.dump(report, f)
        
        results.append(report)
    
    # Or save as JSONL
    with open("./data_json/batch.jsonl", 'w') as f:
        for report in results:
            f.write(json.dumps(report) + '\n')

# Usage
addresses = ["0x1234...", "0x5678...", "0xabcd..."]
batch_analyze(addresses)
```

### Pattern 4: API Integration

**Use Case:** Your analyzer is a REST API

```python
# api_integration.py
import requests
import json

def analyze_via_api(contract_address):
    # Call your analyzer API
    response = requests.post(
        "https://your-analyzer.com/api/analyze",
        json={"contract": contract_address}
    )
    
    report = response.json()
    
    # Convert to ML system format
    ml_report = {
        "metadata": report.get('metadata', {}),
        "bytecode": report.get('bytecode', {}),
        "functions": report.get('functions', {}),
        "temporal_analysis": report.get('temporal', {}),
        "economics": report.get('token_data', {}),
        "gas_profiles": report.get('gas', {})
    }
    
    # Save for ML processing
    with open(f"./data_json/{contract_address}.json", 'w') as f:
        json.dump(ml_report, f, indent=2)
    
    return ml_report
```

---

## 🛠️ Module-Specific Integration Examples

### 1. Contract Analyzer (Bytecode Analysis)

**Tools:** Slither, Mythril, Manticore, custom decompilers

```python
# Example: Slither integration
from slither import Slither

def analyze_with_slither(contract_path):
    slither = Slither(contract_path)
    
    # Extract data
    contract = slither.contracts[0]
    
    report = {
        "metadata": {
            "contract_name": contract.name,
            "chain_id": 1
        },
        "bytecode": {
            "size": len(contract.bytecode),
            "complexity_metrics": {
                "cyclomatic_complexity": calculate_complexity(contract),
                "maintainability_index": calculate_maintainability(contract)
            }
        },
        "functions": {
            "total": len(contract.functions),
            "list": [
                {
                    "name": f.name,
                    "visibility": f.visibility
                }
                for f in contract.functions
            ]
        },
        "temporal_analysis": {},  # Not available from Slither
        "economics": {},          # Not available from Slither
        "gas_profiles": {}        # Not available from Slither
    }
    
    return report
```

### 2. Token Detector (Economics Analysis)

**Tools:** DEX APIs, The Graph, custom token analyzers

```python
# Example: Using Web3.py + Uniswap API
from web3 import Web3

def analyze_token_economics(token_address):
    w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/...'))
    
    # Get token info
    token_contract = w3.eth.contract(
        address=token_address,
        abi=ERC20_ABI
    )
    
    # Get liquidity from Uniswap
    tvl_usd = get_uniswap_liquidity(token_address)
    
    report = {
        "metadata": {
            "chain_id": 1,
            "contract_address": token_address
        },
        "bytecode": {},       # Not focus of this module
        "functions": {},      # Not focus of this module
        "temporal_analysis": {},  # Not focus of this module
        "economics": {
            "total_value_locked_usd": tvl_usd,
            "tokens": get_token_balances(token_address),
            "token_count": len(get_token_balances(token_address))
        },
        "gas_profiles": {}    # Not focus of this module
    }
    
    return report
```

### 3. Activity Tracker (Temporal Analysis)

**Tools:** Etherscan API, The Graph, custom indexers

```python
# Example: Etherscan API integration
import requests

def analyze_contract_activity(contract_address, api_key):
    # Get transaction history
    url = f"https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": contract_address,
        "apikey": api_key
    }
    
    response = requests.get(url, params=params)
    txs = response.json()['result']
    
    # Analyze activity
    unique_users = len(set(tx['from'] for tx in txs))
    recent_txs = [tx for tx in txs if is_recent(tx, days=30)]
    
    report = {
        "metadata": {
            "contract_address": contract_address,
            "chain_id": 1
        },
        "bytecode": {},
        "functions": {},
        "temporal_analysis": {
            "last_interaction_days": days_since_last_tx(txs),
            "unique_users_30d": len(set(tx['from'] for tx in recent_txs)),
            "total_transactions": len(txs),
            "activity_pattern": classify_activity(recent_txs)
        },
        "economics": {},
        "gas_profiles": {
            "average_tx_cost": average_gas(txs),
            "median_tx_cost": median_gas(txs)
        }
    }
    
    return report
```

### 4. Simulator/Executor (Gas Profiling)

**Tools:** Tenderly, Hardhat, custom simulation

```python
# Example: Tenderly simulation
import requests

def simulate_transactions(contract_address):
    # Simulate common operations
    simulations = []
    
    for operation in ['transfer', 'approve', 'swap']:
        result = tenderly_simulate(
            contract_address,
            operation
        )
        simulations.append(result)
    
    # Analyze gas usage
    gas_costs = [sim['gas_used'] for sim in simulations]
    
    report = {
        "metadata": {
            "contract_address": contract_address
        },
        "bytecode": {},
        "functions": {},
        "temporal_analysis": {},
        "economics": {},
        "gas_profiles": {
            "average_tx_cost": sum(gas_costs) / len(gas_costs),
            "max_tx_cost": max(gas_costs),
            "gas_limits": {
                "safe_execution_limit": 3000000,
                "frontrun_protection_required": detect_frontrun_risk(simulations)
            }
        }
    }
    
    return report
```

### 5. Decompiler (Function Analysis)

**Tools:** Panoramix, Heimdall, Etherscan

```python
# Example: Using 4byte directory for function signatures
import requests

def analyze_functions(bytecode):
    # Extract function selectors from bytecode
    selectors = extract_selectors(bytecode)
    
    # Lookup known signatures
    known_functions = []
    unknown_selectors = []
    
    for selector in selectors:
        signature = lookup_4byte(selector)
        if signature:
            known_functions.append({
                "name": signature,
                "selector": selector
            })
        else:
            unknown_selectors.append(selector)
    
    report = {
        "metadata": {},
        "bytecode": {
            "size": len(bytecode)
        },
        "functions": {
            "total": len(selectors),
            "known": len(known_functions),
            "unknown": len(unknown_selectors),
            "list": known_functions
        },
        "temporal_analysis": {},
        "economics": {},
        "gas_profiles": {}
    }
    
    return report

def lookup_4byte(selector):
    """Query 4byte.directory for function signature"""
    response = requests.get(
        f"https://www.4byte.directory/api/v1/signatures/?hex_signature={selector}"
    )
    results = response.json()['results']
    return results[0]['text_signature'] if results else None
```

### 6. Compliance Checker

**Tools:** Chainalysis, Elliptic, custom OFAC checks

```python
# Example: Compliance integration
def check_compliance(contract_address):
    # Check against sanctions lists
    is_sanctioned = check_ofac_list(contract_address)
    
    # Check deployer
    deployer = get_deployer(contract_address)
    deployer_sanctioned = check_ofac_list(deployer)
    
    # Add to report metadata
    report = {
        "metadata": {
            "contract_address": contract_address,
            "compliance": {
                "sanctioned": is_sanctioned,
                "deployer_sanctioned": deployer_sanctioned,
                "risk_score": calculate_compliance_risk(
                    is_sanctioned,
                    deployer_sanctioned
                )
            }
        },
        "bytecode": {},
        "functions": {},
        "temporal_analysis": {},
        "economics": {},
        "gas_profiles": {}
    }
    
    return report
```

---

## 🔄 Complete Integration Pipeline

### End-to-End Example

```python
# complete_pipeline.py
"""
Complete security analysis pipeline combining all modules
"""

import json
from pathlib import Path

# Import your modules
from contract_analyzer import analyze_bytecode
from token_detector import analyze_economics
from activity_tracker import analyze_activity
from gas_profiler import analyze_gas
from decompiler import analyze_functions
from compliance_checker import check_compliance

# Import ML system
from ML_UNIFIED_SYSTEM_V3_3 import MLScorer

def full_security_scan(contract_address):
    """
    Complete security analysis with all modules
    """
    print(f"🔍 Analyzing {contract_address}...")
    
    # 1. Collect data from all modules
    print("  📊 Running bytecode analysis...")
    bytecode_data = analyze_bytecode(contract_address)
    
    print("  💰 Running economics analysis...")
    economics_data = analyze_economics(contract_address)
    
    print("  📈 Running activity analysis...")
    activity_data = analyze_activity(contract_address)
    
    print("  ⛽ Running gas profiling...")
    gas_data = analyze_gas(contract_address)
    
    print("  🔓 Running function analysis...")
    function_data = analyze_functions(contract_address)
    
    print("  ✅ Running compliance check...")
    compliance_data = check_compliance(contract_address)
    
    # 2. Merge all data into unified report
    unified_report = {
        "metadata": {
            **bytecode_data.get('metadata', {}),
            **compliance_data.get('metadata', {}),
            "contract_address": contract_address,
            "analysis_timestamp": datetime.now().isoformat()
        },
        "bytecode": {
            **bytecode_data.get('bytecode', {}),
            **function_data.get('bytecode', {})
        },
        "functions": {
            **bytecode_data.get('functions', {}),
            **function_data.get('functions', {})
        },
        "temporal_analysis": activity_data.get('temporal_analysis', {}),
        "economics": economics_data.get('economics', {}),
        "gas_profiles": gas_data.get('gas_profiles', {})
    }
    
    # 3. Save unified report
    output_path = Path(f"./data_json/{contract_address}.json")
    with open(output_path, 'w') as f:
        json.dump(unified_report, f, indent=2)
    
    print(f"  💾 Report saved: {output_path}")
    
    # 4. ML risk assessment
    print("  🧠 Running ML risk assessment...")
    scorer = MLScorer()
    ml_result = scorer.score_single(unified_report)
    
    # 5. Generate final report
    final_report = {
        "contract_address": contract_address,
        "analysis_timestamp": unified_report['metadata']['analysis_timestamp'],
        "raw_data": unified_report,
        "ml_prediction": ml_result,
        "risk_assessment": {
            "level": ml_result['risk_level'],
            "probability": ml_result['probability'],
            "classification": ml_result['prediction'],
            "confidence": ml_result['confidence']
        },
        "recommendations": generate_recommendations(ml_result)
    }
    
    # 6. Save final report
    final_path = Path(f"./ml_output/{contract_address}_final.json")
    with open(final_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Risk: {ml_result['prediction']} ({ml_result['probability']:.1%})")
    print(f"   Report: {final_path}")
    
    return final_report

def generate_recommendations(ml_result):
    """Generate actionable recommendations"""
    prob = ml_result['probability']
    
    if prob < 0.3:
        return [
            "Contract appears safe based on current analysis",
            "Always verify liquidity and trading functionality",
            "Monitor for changes in behavior"
        ]
    elif prob < 0.7:
        return [
            "Contract shows suspicious patterns - investigate further",
            "Test with small amounts first",
            "Verify contract source code if available",
            "Check community feedback and audit reports"
        ]
    else:
        return [
            "⚠️ HIGH RISK - DO NOT INTERACT",
            "Strong honeypot indicators detected",
            "Avoid trading or providing liquidity",
            "Report to community if not already known"
        ]

# Usage
if __name__ == "__main__":
    # Analyze single contract
    contract = "0x1234567890abcdef..."
    result = full_security_scan(contract)
    
    # Or batch analyze
    contracts = [
        "0x1234...",
        "0x5678...",
        "0xabcd..."
    ]
    
    for contract in contracts:
        result = full_security_scan(contract)
```

---

## 📊 Data Format Conversion

### If Your Format Is Different

```python
# adapter.py
"""
Adapt your analyzer output to ML system format
"""

def adapt_to_ml_format(your_format):
    """
    Convert your analyzer's format to ML system format
    """
    
    # Map your fields to ML system fields
    ml_format = {
        "metadata": {
            "chain_id": your_format.get('network_id', 1),
            "contract_address": your_format.get('address'),
            "deployment_info": {
                "deployment_age_days": calculate_age(
                    your_format.get('deployed_at')
                )
            }
        },
        
        "bytecode": {
            "size": your_format.get('code_size'),
            "complexity_metrics": {
                "cyclomatic_complexity": your_format.get('complexity'),
                "halstead_volume": your_format.get('halstead'),
                "maintainability_index": your_format.get('maintainability'),
                "opcode_diversity": your_format.get('opcode_diversity')
            },
            "runtime_hash": your_format.get('code_hash')
        },
        
        "functions": {
            "total": len(your_format.get('methods', [])),
            "known": count_known(your_format.get('methods', [])),
            "unknown": count_unknown(your_format.get('methods', [])),
            "list": adapt_functions(your_format.get('methods', []))
        },
        
        "temporal_analysis": {
            "last_interaction_days": your_format.get('days_since_last_tx'),
            "unique_users_30d": your_format.get('unique_callers'),
            "activity_pattern": classify_pattern(your_format)
        },
        
        "economics": {
            "total_value_locked_usd": your_format.get('tvl'),
            "tokens": adapt_tokens(your_format.get('tokens', [])),
            "token_count": len(your_format.get('tokens', []))
        },
        
        "gas_profiles": {
            "average_tx_cost": your_format.get('avg_gas'),
            "gas_limits": {
                "safe_execution_limit": your_format.get('gas_limit', 3000000),
                "frontrun_protection_required": your_format.get('frontrun_risk', False)
            }
        }
    }
    
    return ml_format
```

---

## ✅ Integration Checklist

Before deploying your integration:

- [ ] **JSON Schema**: Follows the required schema
- [ ] **Data Types**: All values are correct types (numbers, strings, arrays)
- [ ] **Missing Data**: Handles missing fields gracefully
- [ ] **Error Handling**: Catches and logs analyzer errors
- [ ] **File Naming**: Uses contract address in filename
- [ ] **File Location**: Saves to `./data_json/`
- [ ] **Testing**: Tested with at least 10 contracts
- [ ] **Documentation**: Internal docs for your team
- [ ] **Monitoring**: Logs analyzer success/failure rates
- [ ] **Validation**: Validates output before saving

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Missing Features**
```
ERROR: Feature 'bytecode_size' not found
```
**Solution:** Ensure all required keys exist, use `{}` for unavailable data:
```python
report = {
    "bytecode": {},  # Empty but present
    ...
}
```

**Issue 2: Wrong Data Types**
```
ERROR: Expected float, got string
```
**Solution:** Convert data types:
```python
report = {
    "bytecode": {
        "size": int(your_data.get('size', 0))  # Ensure int
    }
}
```

**Issue 3: Nested Structure**
```
ERROR: Cannot find 'complexity_metrics'
```
**Solution:** Ensure proper nesting:
```python
report = {
    "bytecode": {
        "complexity_metrics": {  # Nested correctly
            "cyclomatic_complexity": 42
        }
    }
}
```

---

## 📞 Support

Need help integrating your analyzer?

- 📖 Check [ARCHITECTURE.md](ARCHITECTURE.md) for feature details
- 💬 Open a GitHub issue with your integration question
- 📧 Email with sample data format

---

**Last Updated:** 2026-02-08  
**Version:** 3.3  
**Maintained by:** ML Unified System Team

