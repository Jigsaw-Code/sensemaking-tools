# To build the docker container
Run all the following commands from the root directory
### Build and Run your service:
`docker compose up --build -d`
### Find the container name:
`docker ps`

Should look something like:
```
CONTAINER ID   IMAGE                             COMMAND               CREATED         STATUS         PORTS     NAMES
7b2de3be0087   participation-project-napolitan   "tail -f /dev/null"   5 seconds ago   Up 4 seconds             participation-project-napolitan-1
```

### Execute a bash shell inside the container:
Use the name from above.\
`docker exec -it participation-project-napolitan-1 bash`

### Setup gcloud cli project and authenticate:
`gcloud auth application-default login && \`\
`gcloud config set project <project_name>`

# To delete docker container
`docker compose down --rmi all --volumes`
