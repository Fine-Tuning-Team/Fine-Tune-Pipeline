# Choreo setup

Before running the Fine-Tune Pipeline, you need to ensure the MLFlow server running in Choreo as well as the MLFLow database is running in choreo.

## MLFlow database setup

!!! tip "Already have a database running?"
    If you already have a database running in Choreo, you can skip this step.

### 1. Deploy the database in Choreo

- Go to [Choreo](https://console.choreo.dev)
- Go to your organization
- Go to `Resources` > `Databases`
- Create a new database
- Choose `PostgreSQL` as the database type
- Deploy the database

### 2. Power on the database

- Go to the database you just created
- Click on `Power On Servicec` to start the database
- Copy the configuration details (host, port, username, password, database name)

## MLFlow server setup

### 1. Deploy the server in Choreo

!!! tip "Already have a server running?"
    If you already have an MLFlow server running in Choreo, you can skip this step.

- Go to [Choreo](https://console.choreo.dev)
- Create a project under your organization
- Deploy the MLFlow server within the project as a web application component
- Use this GitHub repository as the source: [MLFlow Server](https://github.com/Fine-Tuning-Team/MLFlow-Server)

### 2. Configure the MLFlow server

- Go to the MLFlow server component
- Go to `DevOps` > `Configs & Secrets`
- Create a new configuration with the details you found in the MLflow database setup:

```env
    DATABASE = "your-database-name"
    USER = "your-database-username"
    PASSWORD = "your-database-password"
    HOST = "your-database-host"
    PORT = "your-database-port"
```

### 3. Add storage mounts

- Go to `DevOps` > `Storage`
- Create a new storage mount with in-memory type
- Add mount path as `mlruns`

### 4. Configure container entrypoint

- Go to `DevOps` > `Containers`
- Set the command to `["mlflow", "run"]`
- Set the arguments to `["--backend-store-uri", "postgresql+psycopg2://$(USER):$(PASSWORD)@$(HOST):$(PORT)/$(DATABASE)", "--host", "0.0.0.0", "--port", "5000"]`

### 5. Start the MLFlow server

- Go to `Build` and build the latest
- Go to 'Deploy' and deploy the latest build
- Go to the web app url in the `Deploy` tab and ensure the server is running. 

🚀 Great. Now you can move into fine tuning.

## Next Steps

With your environment set up, you're ready to:

1. [Run your first fine-tuning job](quick-start.md)
2. [Explore configuration options](../configuration/overview.md)
3. [Learn about advanced features](../tutorials/advanced-configuration.md)
