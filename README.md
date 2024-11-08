# Participation Project

This repo is meant to help with sensemaking for voting tasks. Comment and vote data can be summarized and categorized.

## Setup

First make sure you have `npm` installed (`apt-get install npm` on Ubuntu-esque systems).

Next install the project modules by running

```
npm install
```

### GCloud setup

- For installation: <https://cloud.google.com/sdk/docs/install-sdk#deb>

```
sudo apt install -y google-cloud-cli

gcloud config set project <your project name here>
gcloud auth application-default login
```

# Development

### Testing

Unit tests can be run with the following command:

```
npm test
```

