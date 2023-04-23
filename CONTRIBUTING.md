# Contributing to Polysplit

First of all, thank you for your interest in contributing to Polysplit! We appreciate your effort and look forward to working together to improve the project. This document provides guidelines and instructions for contributing to the project.

## Prerequisites

Before you can start contributing to Polysplit, you'll need to set up your development environment. Make sure you have the following installed:

- Python 3.7 or later
- Git

## Getting Started

1. Fork the repository on GitHub.

2. Clone your fork of the repository:

```bash
git clone https://github.com/r1p71d3/polysplit.git
cd polysplit
```

3. Set up a new virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

4. Install the development dependencies:

```bash
pip install -r requirements.txt
```

## Before Submitting a Pull Request

Before submitting a pull request, please make sure you've completed the following steps:

1. Ensure your changes follow the code style guidelines by running the following commands:

```bash
flake8 .
black --check .
```

If any style issues are detected, you can automatically fix them with:

```bash
black .
```

2. Run the tests and check for code coverage:

```bash
pytest --cov=polysplit
```

Please add tests for any new features you've implemented or modify the existing tests if needed.

3. Update the documentation if you've added new features or made changes to the existing API.
4. Make sure your commits have clear and descriptive messages.

## Submitting a Pull Request

1. Push your changes to your fork:

```bash
git push
```

2. Create a new pull request on GitHub and provide a clear description of your changes.

## Additional Information

- Please don't include any unrelated changes in your pull request.
- If you're working on a large feature or change, consider creating an issue to discuss it before starting the implementation.

Thank you for contributing to Polysplit!
