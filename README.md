# Solar Challenge : Cross-Country Solar Farm Analysis
Challenge to Kickstart my AI Mastery with Cross-Country Solar Farm Analysis

## 🚀 Environment Setup

To reproduce the development environment, please follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Teshager21/solar-challenge-week1.git
    cd  solar-challenge-week1
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This will install the required Python packages, including `numpy` and `pandas`, as specified in the `requirements.txt` file.

4.  **Set up pre-commit (Optional):**
    ```bash
    pip install pre-commit
    pre-commit install
    ```

    This will install and set up pre-commit to run the checks before committing

## 📁 Project Structure

    📦 TheProjectRoot/
    ├── .vscode/                  # Visual Studio Code settings
    │   └── settings.json
    ├── .github/                 # GitHub Actions workflows
    │   └── workflows/
    │       └── unittests.yml
    ├── .gitignore               # Git ignored files
    ├── requirements.txt         # Python dependencies
    ├── README.md                # Project overview and setup instructions
    ├── data/                    # Input/output datasets or temporary data files
    ├── src/                     # Main application source code
    ├── notebooks/               # Jupyter notebooks for experiments or analysis
    │   ├── __init__.py
    │   └── README.md
    ├── tests/                   # Unit and integration tests
    │   └── __init__.py
    └── scripts/                 # Utility scripts or CLI tools
        ├── __init__.py
        └── README.md
