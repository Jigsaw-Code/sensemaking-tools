# Use a Node.js 18 image (smaller Alpine variant for production)
FROM node:18-alpine

# Since the web-ui imports some types (VoteTally, Comment) from the library,
# we include the dependent source code directly
# to be able to develop off of whatever the current state of the repo is.
COPY library/src/types.ts ./app/library/src/types.ts

# Install dependencies and copy the rest of the application code
WORKDIR /app/web-ui
COPY web-ui/package.json web-ui/package-lock.json ./
RUN npm install
COPY web-ui .

# Prebuild the app to save time during startup
RUN npm run build

# Expose the port the web-ui listens on
EXPOSE 4200

# Define the command to run the web-ui
CMD ["npm", "run", "start:web-ui"]

# TODO: There's a way to make it so that changes to web-ui directory pass through to the container
# as CMD is running, so that you can actually use this as a development environment.
# You can do some of this with docker compose, and there may now be ways of doing it directly
# with docker run via argument flags.
