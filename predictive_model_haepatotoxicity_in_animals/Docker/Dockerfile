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

# Install default precisionFDA Ubuntu packages
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y --no-install-recommends \
        aria2 \
        byobu \
        cmake \
        cpanminus \
        curl \
        dstat \
        g++ \
        git \
        htop \
        libarchive-dev \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libncurses5-dev \
        make \
        python3-dev \
        python3-pip \
        r-base \
        wget \
        xz-utils

# Install default precisionFDA python packages
RUN pip install --break-system-packages \
        requests==2.5.0 \
        futures==2.2.0 \
        setuptools==10.2

# Install python tabstar and other dependencies
RUN pip install --break-system-packages \
                pandas \
                tabstar \
                split-folders==0.5.1 \
                tensorflow \
                keras --upgrade

RUN pip install --break-system-packages git+https://github.com/cmarkello/TabSTAR.git

# Install R packages
RUN R -e "install.packages(c('archive', 'this.path', 'tseries'), \
                           dependencies=TRUE, \
                           repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('devtools', \
                           dependencies=TRUE)"
RUN R -e "install.packages('remotes', \
                           dependencies=TRUE)"
RUN R -e "remotes::install_github(c('cmarkello/SENDQSAR','trevorld/r-optparse'), \
                           dependencies=TRUE)"

# Update apt-get
RUN DEBIAN_FRONTEND=noninteractive apt-get update

# Download helper executables
RUN curl https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/send_data_train_extraction.R -o /usr/bin/send_data_train_extraction.R --no-same-owner --no-same-permissions
RUN curl https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/send_data_test_extraction.R -o /usr/bin/send_data_test_extraction.R --no-same-owner --no-same-permissions
RUN curl https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/run_tabstar_send.py -o /usr/bin/run_tabstar_send.py --no-same-owner --no-same-permissions

# Create directory /work and set it to $HOME and CWD
RUN mkdir -p /work
ENV HOME=/work
WORKDIR /work

# Set entry point to container
ENTRYPOINT ["/usr/bin/run_tabstar_send.py"]