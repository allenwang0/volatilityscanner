# Volatility Scanner

**Volatility Scanner** is a Python-based application designed to analyze, monitor, and simulate market volatility. It provides a web interface to interact with the underlying volatility engine and simulation tools.

## Features

* **Volatility Engine:** Core logic (`engine.py`) for calculating and processing market volatility metrics.
* **Market Simulation:** A dedicated module (`simulator.py`) to run simulations and backtest scenarios based on volatility data.
* **Web Interface:** An interactive frontend (served via `templates`) to visualize data and control scanner parameters.

## Project Structure

* **`main.py`**: The application entry point. This file likely initializes the web server and handles routing.
* **`engine.py`**: Contains the backend algorithms for scanning and calculating volatility.
* **`simulator.py`**: Handles logic for simulating market conditions or trading strategies.
* **`templates/`**: Directory containing HTML templates for the user interface.
* **`requirements.txt`**: A list of Python dependencies required to run the project.

## Installation

Follow these steps to set up the project locally.

### Prerequisites

* Python 3.x
* pip (Python package manager)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/allenwang0/volatilityscanner.git](https://github.com/allenwang0/volatilityscanner.git)
    cd volatilityscanner
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    * *Windows:*
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    * *macOS/Linux:*
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Start the application:**
    Run the main script to start the server.
    ```bash
    python main.py
    ```

2.  **Access the Web Interface:**
    Once the server is running, open your web browser and navigate to the local address shown in your terminal (typically `http://127.0.0.1:5000` or similar).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source.