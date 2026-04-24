# Sensemaking

Jigsaw Sensemaking Microsite

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project to a remote environment.

### Dependencies

- Node: `22.x.x`

### Setup

- Run `npm install` to install the required dependancies.

## Development

- Run `npm run dev` to spin up Webpack and watch for changes.
- Run `npm run preview` to spin up a local web server.
- Navigate to `http://localhost:8080` once the initial build is complete.

## Deployment

### Pages setup
- Settings -> Environments -> github-pages
- Deployment branches and tags -> Add deployment branch or tag rule
- Add "site" branch
- Settings -> Pages
- Build and deployment Source -> "GitHub Actions"

### Overview
This app is built/deployed using GitHub Actions.

### Deploying "Site" changes
Any new commits "pushed" to the `site` branch will automatically trigger a new build/deployment.
