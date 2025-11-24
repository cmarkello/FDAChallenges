# How to use the dockerfile
# 1) Build image
# - in the the directory with Dockerfile run following
#   docker build -t <image_name> .
# - docker image ls # the image should appear in the list

# 2) Run inside of the container
# - docker run -it --entrypoint /bin/bash <image_name>
# - cd /usr/bin/ # going to the run directory
# - ./run --paramenter_name parameter_value # run script and pass parameters

# 3) Run outside of the container (file input example)
# - docker run -v <path_to_local_folder>:/<path_to_folder_inside_container> <image_name> --input_file_parameter_name /path_to_folder_inside_container/file.txt

# For more information please consult the app export section in the precisionFDA docs

# Start with Ubuntu 24.04  base image
FROM ubuntu:24.04

# Set environment variables to avoid interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Update apt, install system dependencies for R and Python packages, and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common dirmngr wget gnupg apt-transport-https build-essential python3 python3-pip python3-venv && \
    # Add CRAN repository for R 4.4 packages on Ubuntu Noble
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
    add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" && \
    apt-get update && \
    # Install R base and development packages
    apt-get install -y --no-install-recommends r-base r-base-dev git && \
    # Clean up apt caches to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install R packages: remotes and others from CRAN
RUN R -e "install.packages(c('remotes', 'archive', 'this.path'), repos='https://cloud.r-project.org')"

RUN R -e "remotes::install_github(c('cmarkello/SENDQSAR','trevorld/r-optparse'))"

# Set up a Python virtual environment for better dependency management
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install python tabstar and other dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        pandas \
        split-folders==0.5.1 \
        tensorflow \
        gdown \
        keras --upgrade

## THIS INSTALLS 11 GB of stuff!
RUN pip install --no-cache-dir git+https://github.com/cmarkello/TabSTAR.git

# Download helper executables
RUN wget -c https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/send_data_train_extraction.R -o /usr/bin/send_data_train_extraction.R
RUN wget -c https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/send_data_test_extraction.R -o /usr/bin/send_data_test_extraction.R
RUN wget -c https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/run_tabstar_send.py -o /usr/bin/run_tabstar_send.py
RUN chmod ugo+rwx /usr/bin/send_data_train_extraction.R /usr/bin/send_data_test_extraction.R /usr/bin/run_tabstar_send.py


# Create directory /work and set it to $HOME and CWD
RUN mkdir -p /work
ENV HOME=/work
WORKDIR /work
# Download the default model
RUN gdown https://drive.google.com/uc?id=1OtGhqnf40OuVnigx3CCmBaL3C2DEIFde -O ~/


# Set entry point to container
ENTRYPOINT ["/usr/bin/run_tabstar_send.py"]