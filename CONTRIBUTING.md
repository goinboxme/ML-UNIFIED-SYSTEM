# Contributing to ML Unified System

Thank you for your interest in contributing to ML Unified System! 🎉

This document provides guidelines for contributing to the project.

---

## 🎯 Ways to Contribute

### 1. 🐛 Report Bugs

Before reporting a bug:
- Check [existing issues](https://github.com/goinboxme/ml-unified-system/issues) to avoid duplicates
- Verify you're using the latest version
- Test with a minimal dataset

**Good Bug Report Includes:**
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Python version
- Operating system (Desktop/Android)
- Sample data (if applicable)
- Error messages and stack traces
- Screenshots (if relevant)

**Example:**
```markdown
**Bug:** Feature extraction fails on nested arrays

**Environment:**
- Python: 3.9.5
- OS: Android 12 (Termux)
- Version: 3.3.0

**Steps to Reproduce:**
1. Create JSON with nested array: `{"functions": {"list": [[...]]}}`
2. Run feature extraction
3. See error

**Expected:** Handles nested arrays gracefully
**Actual:** Crashes with TypeError

**Error Message:**
```
TypeError: 'list' object is not callable at line 234
```

**Sample Data:**
```json
{
  "functions": {
    "list": [["method1", "method2"]]
  }
}
```
```

### 2. 💡 Suggest Features

Feature suggestions are tracked as GitHub issues.

**Good Feature Suggestion Includes:**
- Clear problem statement
- Proposed solution
- Alternative approaches considered
- Use cases and examples
- Impact on existing functionality

**Example:**
```markdown
**Feature Request:** Support for Binance Smart Chain (BSC)

**Problem:**
Currently only supports Ethereum mainnet. Many honeypots exist on BSC.

**Proposed Solution:**
Add chain_id detection and BSC-specific feature extraction.

**Use Cases:**
- Analyze BSC tokens
- Compare cross-chain patterns
- Expand coverage

**Implementation Ideas:**
1. Add chain_id to metadata
2. BSC-specific gas metrics
3. Update feature extraction logic
```

### 3. 🔧 Submit Pull Requests

We love pull requests! Here's how:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
6. **Push to your fork**
7. **Open a Pull Request**

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- Text editor (VS Code, PyCharm, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ml-unified-system.git
cd ml-unified-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Create test data
mkdir -p data_json data_txt data_db
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ML_UNIFIED_SYSTEM_V3_3

# Run specific test
pytest tests/test_feature_extraction.py
```

---

## 📝 Coding Standards

### Python Style

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/):

```python
# Good
def extract_features(contract_data: dict) -> Dict[str, float]:
    """
    Extract behavioral features from contract report.
    
    Args:
        contract_data: JSON contract report
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    # Implementation
    return features

# Bad
def extract(d):
    f={}
    return f
```

### Code Formatting

Use `black` for automatic formatting:

```bash
# Format all files
black .

# Check formatting without changing
black --check .
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Dict, List, Optional, Any

def score_contract(
    contract_data: Dict[str, Any],
    model_path: Optional[str] = None
) -> Dict[str, float]:
    """Scores a single contract"""
    # Implementation
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(data: List[Dict]) -> RandomForestClassifier:
    """
    Train Random Forest model on contract data.
    
    This function performs the full training pipeline including
    feature extraction, normalization, and model fitting.
    
    Args:
        data: List of contract reports (JSON format)
        
    Returns:
        Trained RandomForestClassifier instance
        
    Raises:
        ValueError: If data is empty or malformed
        
    Example:
        >>> contracts = load_contracts()
        >>> model = train_model(contracts)
        >>> print(f"Trained on {len(contracts)} contracts")
    """
    # Implementation
    pass
```

---

## 🧪 Testing Guidelines

### Writing Tests

```python
# tests/test_feature_extraction.py
import pytest
from ML_UNIFIED_SYSTEM_V3_3 import extract_features_from_json

def test_extract_features_minimal():
    """Test feature extraction with minimal JSON"""
    data = {
        "metadata": {},
        "bytecode": {},
        "functions": {},
        "temporal_analysis": {},
        "economics": {},
        "gas_profiles": {}
    }
    
    features = extract_features_from_json(data)
    
    assert isinstance(features, dict)
    assert len(features) == 28
    assert all(isinstance(v, (int, float)) for v in features.values())

def test_extract_features_full():
    """Test with complete JSON"""
    data = {
        "metadata": {"chain_id": 1},
        "bytecode": {"size": 12000, "complexity_metrics": {...}},
        # ... full structure
    }
    
    features = extract_features_from_json(data)
    
    assert features['chain_id'] == 1
    assert features['bytecode_size'] == 12000

def test_extract_features_malformed():
    """Test error handling with malformed data"""
    data = {"invalid": "structure"}
    
    features = extract_features_from_json(data)
    
    # Should return dict with default values, not crash
    assert isinstance(features, dict)
```

### Test Coverage

Aim for:
- **Core functions:** 90%+ coverage
- **Utility functions:** 80%+ coverage
- **Integration tests:** Cover main workflows

### Test on Multiple Platforms

If your changes affect:
- File I/O → Test on Windows, Linux, Mac
- Data parsing → Test with various JSON structures
- Mobile compatibility → Test on Android (Pydroid/Termux)

---

## 📚 Documentation

### Update Documentation When:

- Adding new features → Update README.md
- Changing API → Update ARCHITECTURE.md
- Affecting integrations → Update MODULE_INTEGRATION.md
- Android-specific changes → Update ANDROID_SETUP.md
- Breaking changes → Update CHANGELOG.md

### Documentation Style

**Be Clear and Concise:**
```markdown
# Good
Extract features from contract JSON report.

# Bad
This function is used to extract the various different 
features that we need from the JSON report that contains
information about the contract...
```

**Use Examples:**
```markdown
# Good
Example:
```python
features = extract_features(contract_data)
print(features['bytecode_size'])  # Output: 12458
```

# Without Example (Less Helpful)
This function extracts features from data.
```

**Update Code Comments:**
```python
# Good
# Extract bytecode size (critical honeypot indicator)
bytecode_size = data.get('bytecode', {}).get('size', 0)

# Bad
# Get size
x = data.get('bytecode', {}).get('size', 0)
```

---

## 🔀 Git Workflow

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format
<type>: <description>

[optional body]

[optional footer]

# Types
feat:     New feature
fix:      Bug fix
docs:     Documentation only
style:    Code style (formatting, semicolons, etc.)
refactor: Code change that neither fixes bug nor adds feature
perf:     Performance improvement
test:     Adding tests
chore:    Build process or auxiliary tool changes
```

**Examples:**

```bash
# Good
feat: add XGBoost model support
fix: resolve feature extraction crash on nested arrays
docs: update Android setup guide for Termux
refactor: simplify feature normalization logic

# Bad
update code
fix bug
changes
```

**Detailed Example:**
```bash
feat: add cross-validation monitoring

Implement 3-fold cross-validation during training to detect
overfitting early. Adds CV F1 score logging and warnings when
train-test gap exceeds threshold.

Closes #42
```

### Branch Naming

```bash
# Feature
feature/xgboost-support
feature/web3-integration

# Bug fix
fix/nested-array-crash
fix/android-termux-compatibility

# Documentation
docs/update-readme
docs/add-examples

# Refactoring
refactor/feature-extraction
refactor/model-training
```

### Pull Request Process

1. **Ensure all tests pass**
   ```bash
   pytest
   black --check .
   flake8 .
   ```

2. **Update documentation**
   - README.md (if feature affects usage)
   - CHANGELOG.md (describe changes)
   - Code comments (if adding complex logic)

3. **Write descriptive PR description**
   ```markdown
   ## Changes
   - Added XGBoost model support
   - Updated feature extraction for new model
   - Added tests for XGBoost integration
   
   ## Testing
   - [x] Tested on Desktop (Linux, Windows, Mac)
   - [x] Tested on Android (Pydroid 3, Termux)
   - [x] All unit tests pass
   - [x] Manual testing with 100+ contracts
   
   ## Screenshots
   [If UI changes]
   
   ## Breaking Changes
   None
   
   ## Related Issues
   Closes #42
   Fixes #38
   ```

4. **Respond to review feedback**
   - Be open to suggestions
   - Ask questions if unclear
   - Make requested changes promptly

5. **Squash commits if requested**
   ```bash
   git rebase -i HEAD~3  # Squash last 3 commits
   ```

---

## 🎯 Contribution Areas

### High Priority

- 🐛 Bug fixes (especially critical security issues)
- 📱 Android compatibility improvements
- 🧪 Test coverage expansion
- 📚 Documentation improvements
- ♿ Accessibility enhancements

### Medium Priority

- ✨ New feature development
- 🎨 UI/UX improvements (if applicable)
- ⚡ Performance optimizations
- 🌍 Multi-chain support

### Nice to Have

- 📊 Visualization improvements
- 🔌 API development
- 🐳 Docker/Kubernetes configs
- 📱 Mobile app development

---

## 🚫 What NOT to Contribute

- Unrelated features outside project scope
- Breaking changes without discussion
- Code without tests
- Undocumented complex logic
- Proprietary or licensed code
- Malicious code or backdoors

---

## 🔒 Security Issues

**DO NOT** open public issues for security vulnerabilities!

Instead:
1. Email: inbox.globaltrade@gmail.com
2. Subject: "[SECURITY] ML Unified System - [Brief Description]"
3. Include:
   - Vulnerability description
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We'll respond within 48 hours.

---

## 📋 Pull Request Checklist

Before submitting:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] No linting errors (`flake8 .`)
- [ ] Type hints added where applicable
- [ ] Docstrings added for new functions
- [ ] Documentation updated (README, CHANGELOG)
- [ ] Tested on multiple platforms (if applicable)
- [ ] No breaking changes (or documented if necessary)
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains changes
- [ ] Related issues referenced

---

## 🏆 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in documentation (for significant contributions)
- Given credit in commit history

Top contributors may be invited as collaborators!

---

## 💬 Communication

### Asking Questions

- 💬 GitHub Discussions (preferred)
- 🐛 GitHub Issues (for bugs only)
- 📧 Email (for private matters)

### Getting Help

Stuck? Ask for help!
- Comment on your PR
- Open a discussion
- Join community chat (if available)

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on project improvement
- Show empathy

**Unacceptable:**
- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private info

---

## 🎓 Learning Resources

New to:
- **Python?** → [Python Tutorial](https://docs.python.org/3/tutorial/)
- **Machine Learning?** → [Scikit-learn Docs](https://scikit-learn.org/)
- **Git?** → [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- **Testing?** → [Pytest Docs](https://docs.pytest.org/)
- **Blockchain?** → [Ethereum Docs](https://ethereum.org/en/developers/docs/)

---

## ❓ Questions?

Have questions about contributing?

- 📖 Read existing [issues](https://github.com/goinboxme/ml-unified-system/issues)
- 💬 Start a [discussion](https://github.com/goinboxme/ml-unified-system/discussions)
- 📧 Email: inbox.globaltrade@gmail.com

---

## 🙏 Thank You!

Every contribution helps:
- Bug reports improve stability
- Feature requests guide development
- Documentation helps users
- Code contributions add value
- Community support encourages others

**Together, we make DeFi safer!** 🛡️

---

**Last Updated:** 2026-02-08  
**Maintained by:** goinboxme  
**Version:** 3.3.0

