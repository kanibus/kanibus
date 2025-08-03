# 🤝 Contributing to Kanibus

Thank you for your interest in contributing to Kanibus! We welcome contributions from the community and are excited to see what you can bring to this project.

---

## 📋 **Table of Contents**

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [How to Contribute](#-how-to-contribute)
- [Bug Reports](#-bug-reports)
- [Feature Requests](#-feature-requests)
- [Pull Requests](#-pull-requests)
- [Code Style](#-code-style)
- [Testing](#-testing)
- [Documentation](#-documentation)

---

## 📜 **Code of Conduct**

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code:

- **Be respectful** and inclusive
- **Be constructive** in discussions and feedback
- **Focus on the issue**, not the person
- **Help create a positive environment** for everyone

---

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8 or higher
- Git for version control
- Basic understanding of ComfyUI and PyTorch
- Familiarity with computer vision concepts (helpful but not required)

### **Areas where we need help**
- 🐛 **Bug fixes** - Help us squash bugs and improve stability
- ✨ **New features** - Implement new nodes or enhance existing ones
- 📚 **Documentation** - Improve docs, tutorials, and examples
- 🧪 **Testing** - Write tests and improve test coverage
- 🎨 **UI/UX** - Improve node interfaces and user experience
- 🌍 **Localization** - Translate documentation and interfaces

---

## 💻 **Development Setup**

### **1. Fork and Clone**
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/kanibus.git
cd kanibus

# Add the original repository as upstream
git remote add upstream https://github.com/kanibus/kanibus.git
```

### **2. Create Development Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Install in development mode
pip install -e .
```

### **3. Set up Pre-commit Hooks**
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

### **4. Run Tests**
```bash
# Run the test suite
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov=nodes --cov-report=html
```

---

## 🛠️ **How to Contribute**

### **1. Choose an Issue**
- Browse [open issues](https://github.com/kanibus/kanibus/issues)
- Look for `good first issue` or `help wanted` labels
- Comment on the issue to let others know you're working on it

### **2. Create a Branch**
```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### **3. Make Changes**
- Write clean, documented code
- Follow our coding standards
- Add tests for new functionality
- Update documentation as needed

### **4. Test Your Changes**
```bash
# Run tests
python -m pytest tests/

# Test with ComfyUI integration
# (Manual testing instructions)

# Check code style
flake8 src/ nodes/
black --check src/ nodes/
```

### **5. Submit Pull Request**
- Push your branch to your fork
- Create a pull request with a clear title and description
- Link any related issues
- Be responsive to feedback

---

## 🐛 **Bug Reports**

When reporting bugs, please include:

### **Required Information**
- **Description**: Clear description of the bug
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, GPU details, ComfyUI version

### **Optional but Helpful**
- Screenshots or videos
- Log files from `logs/kanibus.log`
- Minimal example workflow that demonstrates the bug
- Error messages and stack traces

### **Bug Report Template**
```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g. Windows 11, macOS 13, Ubuntu 22.04]
- Python: [e.g. 3.10.5]
- GPU: [e.g. RTX 4090, Apple M2]
- ComfyUI Version: [e.g. latest, commit hash]
- Kanibus Version: [e.g. v1.0.0]

## Additional Context
Add any other context about the problem here
```

---

## ✨ **Feature Requests**

We love feature requests! When suggesting new features:

### **Good Feature Requests Include**
- **Clear use case**: Why is this feature needed?
- **Detailed description**: What should the feature do?
- **Examples**: How would users interact with it?
- **Implementation ideas**: Any thoughts on how to implement it?

### **Feature Request Template**
```markdown
## Feature Summary
Brief description of the feature

## Use Case
Why do you need this feature? What problem does it solve?

## Detailed Description
Detailed description of what the feature should do

## Example Usage
How would users interact with this feature?

## Implementation Ideas
Any thoughts on how this could be implemented?

## Additional Context
Any other context or screenshots about the feature request
```

---

## 🔄 **Pull Requests**

### **Pull Request Guidelines**
- **Clear title**: Summarize what the PR does
- **Detailed description**: Explain the changes and why they're needed
- **Link issues**: Reference any related issues
- **Test coverage**: Include tests for new functionality
- **Documentation**: Update docs for user-facing changes

### **Pull Request Template**
```markdown
## Description
Brief description of the changes

## Changes Made
- [ ] Added new feature X
- [ ] Fixed bug Y
- [ ] Updated documentation Z

## Testing
- [ ] All existing tests pass
- [ ] Added tests for new functionality
- [ ] Manually tested with ComfyUI

## Documentation
- [ ] Updated README if needed
- [ ] Updated node documentation
- [ ] Added examples if applicable

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] No breaking changes (or clearly documented)
- [ ] Linked related issues

## Screenshots (if applicable)
Add screenshots to help explain your changes
```

---

## 🎨 **Code Style**

### **Python Style Guide**
- **PEP 8**: Follow Python PEP 8 style guide
- **Black**: Use Black formatter with default settings
- **Flake8**: Use Flake8 for linting
- **Type hints**: Use type hints for all public APIs
- **Docstrings**: Google-style docstrings for all functions and classes

### **Code Formatting**
```bash
# Format code with Black
black src/ nodes/

# Check with Flake8
flake8 src/ nodes/

# Sort imports
isort src/ nodes/
```

### **Naming Conventions**
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### **Documentation Standards**
```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.

    Longer description if needed. Explain what the function does,
    any important behaviors, and how to use it.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 0)

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this exception is raised
        TypeError: Description of when this exception is raised

    Example:
        >>> result = example_function("hello", 42)
        >>> print(result)
        True
    """
    # Implementation here
    return True
```

---

## 🧪 **Testing**

### **Test Requirements**
- **Coverage**: Aim for 90%+ test coverage
- **Types**: Unit tests, integration tests, and manual testing
- **Framework**: Use pytest for all tests
- **Structure**: Mirror the source structure in tests/

### **Writing Tests**
```python
import pytest
from unittest.mock import Mock, patch
from src.neural_engine import NeuralEngine

class TestNeuralEngine:
    def test_initialization(self):
        """Test engine initializes correctly."""
        engine = NeuralEngine()
        assert engine is not None
        assert engine.device is not None

    def test_processing_with_valid_input(self):
        """Test processing with valid input."""
        engine = NeuralEngine()
        result = engine.process(valid_input)
        assert result is not None
        assert result.success is True

    @patch('src.neural_engine.torch.cuda.is_available')
    def test_gpu_detection(self, mock_cuda):
        """Test GPU detection logic."""
        mock_cuda.return_value = True
        engine = NeuralEngine()
        assert engine.device.type == 'cuda'
```

### **Running Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=nodes --cov-report=html

# Run specific test file
pytest tests/test_neural_engine.py

# Run specific test
pytest tests/test_neural_engine.py::TestNeuralEngine::test_initialization
```

---

## 📚 **Documentation**

### **Documentation Types**
- **Code docs**: Docstrings in code
- **User docs**: README, usage guides
- **API docs**: Generated from docstrings
- **Examples**: Example workflows and tutorials

### **Documentation Guidelines**
- **Clear and concise**: Write for your audience
- **Examples**: Include practical examples
- **Up to date**: Keep docs synchronized with code
- **Accessible**: Use clear language and good structure

### **Building Documentation**
```bash
# Generate API documentation
sphinx-build -b html docs/ docs/_build/

# Check for broken links
sphinx-build -b linkcheck docs/ docs/_build/
```

---

## 🏷️ **Release Process**

### **Version Numbering**
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### **Release Checklist**
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Create release tag
- [ ] Create GitHub release
- [ ] Update installation instructions

---

## 💬 **Communication**

### **Where to Ask Questions**
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: staffytech@proton.me for direct contact

### **Response Times**
- **Issues**: We aim to respond within 48 hours
- **Pull Requests**: Initial review within 72 hours
- **Questions**: Response within 24-48 hours

---

## 🙏 **Recognition**

Contributors are recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Major contributions highlighted
- **Documentation**: Attribution for significant docs contributions

---

## 📄 **License**

By contributing to Kanibus, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Kanibus! 🎉**

Your contributions help make eye-tracking technology more accessible and powerful for everyone. Whether you're fixing a typo, adding a feature, or helping with documentation, every contribution matters.

*Happy coding!* 🚀