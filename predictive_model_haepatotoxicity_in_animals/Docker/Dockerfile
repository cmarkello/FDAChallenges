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
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get update && apt-get install -y --no-install-recommends \
        aria2 \
        byobu \
        cmake \
        cpanminus \
        curl \
        dstat \
        gfortran \
        g++ \
        gcc \
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
        xz-utils \
        libssl-dev \
        libsodium-dev \
        libpng-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libfreetype6-dev \
        libfontconfig1-dev \
        libfl-dev \
        libgit2-dev build-essential \
        libbz2-dev liblzma-dev \
        libtiff-dev libjpeg-dev libwebp-dev \
        libblas-dev liblapack-dev


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
                gdown \
                keras --upgrade

RUN pip install --break-system-packages git+https://github.com/cmarkello/TabSTAR.git

# Install R packages
RUN R -e "install.packages(c('archive', 'this.path', 'tseries', 'sendigR', 'randomForest', 'reprtree'), \
                           dependencies=TRUE, \
                           repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('devtools', \
                           dependencies=TRUE)"

RUN R -e "devtools::install_github(c('cmarkello/SENDQSAR','trevorld/r-optparse'), \
                           dependencies=TRUE)"

# Update apt-get
RUN DEBIAN_FRONTEND=noninteractive apt-get update

# Download helper executables
RUN curl https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/send_data_train_extraction.R -o /usr/bin/send_data_train_extraction.R
RUN curl https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/send_data_test_extraction.R -o /usr/bin/send_data_test_extraction.R
RUN curl https://raw.githubusercontent.com/cmarkello/FDAChallenges/refs/heads/main/predictive_model_haepatotoxicity_in_animals/scripts/run_tabstar_send.py -o /usr/bin/run_tabstar_send.py
RUN chmod ugo+rwx /usr/bin/send_data_train_extraction.R /usr/bin/send_data_test_extraction.R /usr/bin/run_tabstar_send.py


# Create directory /work and set it to $HOME and CWD
RUN mkdir -p /work
ENV HOME=/work
WORKDIR /work
# Download the default model
RUN gdown https://drive.google.com/uc?id=1OtGhqnf40OuVnigx3CCmBaL3C2DEIFde -O ~/


# Set entry point to container
ENTRYPOINT ["/usr/bin/run_tabstar_send.py"]