# Contributing Guidelines

We welcome contributions to improve documentation, fix implementation bugs, or add test cases.

## Contribution Guidelines
1.  **Fork the repository** and create a feature branch.
2.  **Ensure all tests pass** before submitting a Pull Request:
    ```bash
    python -m unittest discover tests
    ```
3.  **Strict Architecture Freeze**: Do not modify the machine learning model architecture, hyperparameters, or training logs without prior coordination and an approved experimental issue report.
4.  **Format Code**: Follow PEP 8 guidelines for Python code.
