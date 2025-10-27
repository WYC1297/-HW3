# Project Context

## Purpose
Homework assignment (hw3) implementing a spam email classification system using machine learning. The project demonstrates:
- Machine learning model development (Logistic Regression for classification)
- Interactive web application using Streamlit
- Spec-driven development practices with OpenSpec
- Multi-phase development approach

## Tech Stack
- **Primary Language**: Python 3.x
- **Machine Learning**: scikit-learn (Logistic Regression, SVM)
- **Web Framework**: Streamlit for interactive UI
- **Data Processing**: pandas, numpy
- **Environment Management**: virtualenv
- **Development Framework**: OpenSpec for specification-driven development
- **Version Control**: Git with GitHub integration
- **Documentation**: Markdown-based specifications and change proposals

## Project Conventions

### Code Style
- Follow PEP 8 for Python code formatting
- Use descriptive variable and function names (snake_case for Python)
- Include docstrings for all public functions and classes
- Maintain consistent indentation (4 spaces)
- Type hints for function parameters and return values

### Architecture Patterns
- Spec-first development: All features must have corresponding specifications
- Modular design: Separate data processing, model training, and UI components
- Single responsibility principle: Each module has one clear purpose
- Clear separation between ML pipeline and web interface
- Configuration-driven approach for model parameters

### Testing Strategy
- Unit tests for core business logic
- Integration tests for API endpoints and external dependencies
- Specification compliance tests to ensure implementation matches specs
- Test coverage target: >80% for critical paths

### Git Workflow
- Feature branch workflow: `feature/description` or `change/change-id`
- Commit messages: `type(scope): description` (e.g., `feat(auth): add login endpoint`)
- Pull requests required for all changes
- Squash commits when merging to main

## Domain Context
Spam email classification project demonstrating:
- Machine learning classification techniques (Logistic Regression, SVM)
- Text preprocessing and feature extraction
- Model evaluation and performance metrics
- Interactive web application development
- Multi-phase development methodology
- Data pipeline from raw CSV to trained model

**Project Phases**:
- Phase 0: Environment setup (virtualenv named "HW3")
- Phase 1: Data acquisition and SVM baseline model
- Phase 2-N: Additional phases to be defined

## Important Constraints
- Must follow OpenSpec workflow for all feature changes
- All requirements must include scenarios for validation
- No implementation without approved change proposals
- Educational context: focus on learning ML concepts and best practices
- Data source: SMS spam dataset from GitHub repository
- Environment isolation required using virtualenv

## External Dependencies
- OpenSpec CLI for specification management
- GitHub for version control and collaboration
- SMS spam dataset: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Python ML ecosystem: scikit-learn, pandas, numpy, streamlit
