**Create the environment from the requirements.txt file:
export LC_ALL=C
sudo pip install virtualenv      # This may already be installed
virtualenv -p python3 .env       # Create a virtual environment (python3)
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# if the pip install fails due to permission problems, run the following line and then pip install again:
sudo chmod 777 .env -R
deactivate