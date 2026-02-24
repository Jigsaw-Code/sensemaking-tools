# Use a Node.js 18 image (smaller Alpine variant for production)
FROM node:18-alpine

# Copy the application directories and install dependencies for each component
WORKDIR /app/library
COPY library/package.json library/package-lock.json ./
RUN npm install
COPY library/src ./src

WORKDIR /app/api-server
COPY api-server/package.json api-server/package-lock.json ./
RUN npm install
COPY api-server/ .

# Pre-build the api-server to reduce startup time.
# Since 'npm run build' is included in 'npm start',
# this step will be skipped if the build already exists.
RUN npm run build

# Expose the port the api-server listens on
EXPOSE 3000

# Define the command to run the api-server
CMD ["npm", "start"]
