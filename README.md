# Build and Deploy Machine Learning Models on the Cloud


### Instructions to build and deploy on Digital Ocean

1. Login to [DigitalOcean](https://www.digitalocean.com/)  
2. Create Droplet  
3. Click on **one-click apps** and select **Machine Learning and AI**  
4. Choose size (2GB/2CPU is a good start)  
5. Choose a data center region  
6. Add SSH keys  
7. Give name to the host and click create  

Give it a few seconds and the droplet will up. Copy the IP address of the droplet

Now, come to the terminal and type the following command

    $ ssh science@public-ip-of-droplet

Here, remember the name of the jupyter notebook provided in the welcome message. This is what we need to login to the jupyter notebook on Digital Ocean.

But before we get there, we need the repository.

Now, clone this repository

    $ git clone https://github.com/amitkaps/datascience.git

The first step is to create a virtual environment

    $ cd datascience
    $ virtualenv mlcloud   # mlcloud is the name of the virtual environment

    $ source mlcoud/bin/activate    # activate the virtual environment

Now, the required libraries need to be installed.

    $ pip install -r requirements.txt

Now, the virtual environment needs to be activated for `Jupyter Notebook`

    $ python -m ipykernel install --user --name=mlcloud


We are all set. Now, open the jupyter notebook by entering the URL provided in the welcome message, in the browser



