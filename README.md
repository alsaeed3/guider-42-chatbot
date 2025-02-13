# 42 Abu Dhabi AI Guide Chatbot

An AI-powered chatbot specifically designed to answer questions about 42 Abu Dhabi coding school. Built with TinyLlama and optimized for both desktop and Raspberry Pi environments.

## Description

This chatbot provides accurate information about:
- Admission process and Piscine
- Curriculum and projects  
- Campus facilities
- Student life
- Prerequisites and requirements
- Application deadlines
- School policies
- Available resources

## Installation & Setup

### Linux/macOS

1. Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```

2. Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```

3. Clean pip cache:
    ```bash
    pip3 cache purge
    ```

4. Upgrade pip:
    ```bash
    pip3 install --upgrade pip
    ```

5. Install required packages:
    ```bash
    pip3 install -r requirements.txt
    ```

6. Run the chatbot:
    ```bash
    python3 guider.py
    ```

### Windows

1. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:
    ```bash
    .\venv\Scripts\activate
    ```

3. Clean pip cache:
    ```bash
    pip cache purge
    ```

4. Upgrade pip:
    ```bash
    pip install --upgrade pip
    ```

5. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

6. Run the chatbot:
    ```bash
    python guider.py
    ```

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection for initial model download

## Features

- 🎯 Focused responses about 42 Abu Dhabi
- 💾 Memory-efficient implementation
- ⚡ Response caching for common questions
- 🛡️ Comprehensive error handling
- 🔄 CPU/GPU compatibility

## Troubleshooting

### SSL Certificate Issues (Raspberry Pi)
If you encounter SSL certificate errors, use:
    ```bash
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
    ```

### Memory Issues
- Close other applications to free up memory
- Ensure you have at least 4GB of available RAM

### Model Download Issues
- Check your internet connection
- Ensure you have sufficient disk space (at least 2GB free)

## Project Structure
├── README.md
├── requirements.txt
└── src
    ├── cache_utils.py
    └── guider.py

## License

This project is open source and available under the MIT License.

## Acknowledgments

- 42 Abu Dhabi
- Hugging Face
- TinyLlama team