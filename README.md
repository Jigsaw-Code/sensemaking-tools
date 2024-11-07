# Participation Project

This repo is meant to help with sensemaking for voting tasks. Comment and vote data can be summarized and categorized.

## Setup

First make sure you have `npm` installed (`apt-get install npm` on Ubuntu-esque systems).

Next install the project modules by running

```
npm install
```

### GCloud setup

* For external installation: <https://cloud.google.com/sdk/docs/install-sdk#deb>
* For Google internal: <go/gcloudcli>

```
sudo apt install -y google-cloud-cli

gcloud config set project <your project name here>
gcloud auth application-default login
```
