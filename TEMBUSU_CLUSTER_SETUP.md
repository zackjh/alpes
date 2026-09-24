# Tembusu Cluster Setup Guide

This guide sets up ALPES on the SoC Tembusu compute cluster for F3Set and F3ED
active-learning experiments. The commands below assume Bash.

## Prerequisites

1. Create a [SoC Unix account](https://mysoc.nus.edu.sg/~newacct/) if you do not
   already have one.
2. Enable SoC Compute Cluster access on the
   [account services page](https://mysoc.nus.edu.sg/~myacct/services.cgi).
3. Confirm that you can connect through the SoC SSH jump hosts using the
   [SSH jump host quick guide](https://dochub.comp.nus.edu.sg/cf/guides/sjump/quick_guide).

## 1. Connect to the cluster

Run this command on your **local machine**, replacing both occurrences of
`<username>` with your SoC Unix username:

```sh
ssh -J <username>@stujump.comp.nus.edu.sg <username>@xlogin.comp.nus.edu.sg
```

Run all remaining setup commands in the **cluster SSH session**.

## 2. Clone the repository

```sh
git clone https://github.com/zackjh/alpes.git
cd alpes
```

## 3. Install Miniconda

Skip this step if Conda is already installed and available in your shell.
The following commands use the Linux x86-64 installer and save it in your home
directory:

```sh
curl -fL -o ~/Miniconda3-latest-Linux-x86_64.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

Follow the installer prompts, choose an installation directory, and answer `yes`
when asked to initialize Conda. Then reload your Bash configuration:

```sh
source ~/.bashrc
conda --version
```

See the [official Miniconda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install)
for details and troubleshooting.

## 4. Create the ALPES environment

From the repository root, create and activate the environment, then install ALPES:

```sh
conda env create --file envs/alpes-f3set/environment.yml --name alpes-f3set
conda activate alpes-f3set
python -m pip install -e . --no-deps
```

The environment file manages Python and third-party dependencies. The editable
install makes the repository's source available to Python; `--no-deps` keeps
dependency management in Conda.
