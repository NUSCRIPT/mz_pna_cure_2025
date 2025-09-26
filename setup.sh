# Create a virtual environment
python3 -m venv .pcp

# Activate the virtual environment
source .pcp/bin/activate

# Load all packages from requirements.txt
pip install -r requirements.txt

# Install Jupyter and IPykernel
pip install jupyter ipykernel

# Register the virtual environment as a kernel for Jupyter
python -m ipykernel install --user --name=.pcp --display-name="Python(PCP)"
