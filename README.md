# LanceLLM: Large Language Model from Scratch

> "What I cannot create, I do not understand." — Richard Feynman

Welcome to my implementation of a GPT-style language model built from scratch using PyTorch. This project follows the architecture and training methodology from Sebastian Raschka’s *"Build a Large Language Model (from Scratch)."*

## Getting Started
Follow these steps to set up your local development environment and run the model.

### 1. Clone the Repository
```bash
git clone [https://github.com/Lancelot4306/MyLLM2.git](https://github.com/Lancelot4306/MyLLM2.git)
cd MyLLM2
```

### 2. Prerequisites
Ensure you have Python installed. Verify your installation:

```bash
python --version
```

### 3. Setup Virtual Environment
I use uv for fast package management:

```bash
# Install uv
pip install uv
```

#### Create virtual environment (Python 3.10)
```bash
uv venv --python=python3.10
```

#### Activate it (Windows)
```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
uv pip install packaging
uv pip install -r requirements.txt

# Note: If you run into issues, try re-installing the problematic package specifically: 
uv pip install <package_name>
```

### 5. Verify & Run
Check if your environment is correctly configured:

```bash
python python_environment_check.py
```

### 6. Model Operations
Once configured, you can generate your model weights and launch the UI:

#### Save & Load Model Weights:

```bash
python save_model.py
Run the Chatbot UI:
```

```bash
chainlit run app.py
```

---

## Tech Stack

### Framework

* PyTorch

### UI Framework

* Chainlit

### Package Management

* uv

---

## Credits
This project is inspired by the work of Sebastian Raschka. Check out his official repository for the source material.
