FROM jupyter/minimal-notebook:latest

USER root
RUN apt-get -y update --fix-missing && apt-get -y upgrade
RUN apt-get -y install libgtk-3-0 tmux cm-super dvipng texlive-latex-extra texlive-fonts-recommended
s# Install dependencies
USER jovyan
COPY env.yml env.yml
RUN mamba env update --file env.yml
# RUN python3 -m pip install giotto-tda==0.5.1 kmapper==2.0.1

# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
