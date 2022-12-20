# Databricks notebook source
# MAGIC %md # Setup

# COMMAND ----------

# MAGIC %md ## Preparation: Workspace Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Following the steps below will have the following outcomes:
# MAGIC - A **P**ersonal **A**ccess **T**oken is created in the **remote workspace** (where our remote model registry resides)
# MAGIC - Store the PAT and additional information using **secrets** in the **client workspace** (from which this demo is executed)
# MAGIC 
# MAGIC See the diagram below for reference / context.
# MAGIC 
# MAGIC <img src="https://gist.githubusercontent.com/johnkdb/f935f9e6667ac7600405adc6dfd7e392/raw/6900f6c8d0539d597a1756014801f588818208ff/arch.png" alt="alt text" width="1000"/>
# MAGIC 
# MAGIC It is assumed that the `databricks` CLI has been **configured with profiles** for the remote and client workspace, named "cmr" and "client" below respectively.
# MAGIC 
# MAGIC 1. Create a personal access token in the remote workspace
# MAGIC     ```
# MAGIC     # Store the result in $PAT for later reference in step 3.
# MAGIC     export PAT=$(databricks --profile cmr tokens create --lifetime-seconds -1 --comment 'Remote MLflow access' | jq '.token_value')
# MAGIC     ```
# MAGIC 
# MAGIC 
# MAGIC 2. Create a secret scope in this workspace
# MAGIC     ```
# MAGIC     databricks --profile client secrets create-scope --scope <scope-name>
# MAGIC     ```
# MAGIC 
# MAGIC 3. Create three (3) secrets in the scope above: `<prefix>-host`, `<prefix>-workspace-id`, `<prefix>-token`
# MAGIC     ```
# MAGIC     databricks --profile client secrets put --scope <scope-name> --key <prefix>-host --string-value https://eastus2.azuredatabricks.net/
# MAGIC     databricks --profile client secrets put --scope <scope-name> --key <prefix>-workspace-id --string-value 5206439413157315
# MAGIC     databricks --profile client secrets put --scope <scope-name> --key <prefix>-token --string-value $PAT
# MAGIC     ```

# COMMAND ----------

# MAGIC %md ## Demo Data & Functions

# COMMAND ----------

from pprint import pprint

import matplotlib.pyplot as plt

import mlflow

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


def get_or_create_experiment(experiment_name):
  full_experiment_location = f'/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/{experiment_name}'
  existing_experiments = mlflow.search_experiments(mlflow.entities.ViewType.ACTIVE_ONLY, max_results=1, filter_string=f"name = '{full_experiment_location}'")
  return existing_experiments[0].experiment_id if existing_experiments else mlflow.create_experiment(name=full_experiment_location)


# Pre-trained model to be reused in the different experiments
# ...in the different experiments of this demo.
diabetes = load_diabetes()
trained_model = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3).fit(diabetes.data, diabetes.target)

# Dummy artifact to log in experiment runs
plt.savefig('/tmp/plot.png');

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

# Remote MLflow URI used independently for the Tracking Server *and/or* Model Registry
remote_mlflow_uri = f'databricks://mlops:johnk-eastus2'  # databricks://<secret scope>:<secrets prefix>

registered_model_name = 'johnk-az-rf'

# COMMAND ----------

# MAGIC %md # Experiments
# MAGIC 
# MAGIC **This section performs the following for both the local (this) workspace and the remote workspace.**
# MAGIC 
# MAGIC - Create a new experiment (or reuse an existing by ID)
# MAGIC   - (Optional) Set the experiment artifact_location to a mounted dbfs directory.
# MAGIC - Create a new run inside the experiment.
# MAGIC   - Log model artifact
# MAGIC   - Log file artifact
# MAGIC - Inspect the results
# MAGIC 
# MAGIC **Side note on custom experiment `artifact_uri`s**
# MAGIC - It is possible to specify an experiment `artifact_uri` that points to a DBFS *mount point* in either workspace, but that comes with the same [limitations as for non-DBFS locations](https://learn.microsoft.com/en-us/azure/databricks/mlflow/tracking#--create-workspace-experiment) and cannot be registered in the Model Registry. If such `artifact_uri`s are used, one must first copy the logged model to a non-mount DBFS location before registering the model.

# COMMAND ----------

# MAGIC %md ## Local experiment

# COMMAND ----------

# Create experiment in current workspace
mlflow.set_tracking_uri('databricks')

# Log a model in a new run inside the experiment.
with mlflow.start_run(experiment_id=get_or_create_experiment('mlflow_experiments/local_demo')) as local_run:
  mlflow.sklearn.log_model(trained_model, 'random-forest-model')
  mlflow.log_artifact('plot.png')
  
print(local_run.info)

# COMMAND ----------

# MAGIC %md ## Remote experiment

# COMMAND ----------

# Create experiment in remote workspace
mlflow.set_tracking_uri(remote_mlflow_uri)

# Log a model in a new run inside the remote experiment.
with mlflow.start_run(experiment_id=get_or_create_experiment('mlflow_experiments/remote_demo')) as remote_run:
  mlflow.sklearn.log_model(trained_model, 'random-forest-model')
  mlflow.log_artifact('/tmp/plot.png')
  
print(remote_run.info)

# COMMAND ----------

# MAGIC %md # Remote Model Registry
# MAGIC 
# MAGIC Registering models referencing the remote and local tracking servers.

# COMMAND ----------

mlflow.set_registry_uri(remote_mlflow_uri)

client = mlflow.client.MlflowClient(registry_uri=remote_mlflow_uri)

# COMMAND ----------

# MAGIC %md ## Referencing the *remote* workspace Tracking Server

# COMMAND ----------

# DBTITLE 1,Register new model version
# Register a model from the remote tracking server
mlflow.set_tracking_uri(remote_mlflow_uri)
mlflow.register_model(
  model_uri=f'runs:/{remote_run.info.run_id}/random-forest-model',
  name=registered_model_name)

# COMMAND ----------

# MAGIC %md ### Inspect results in remote model registry
# MAGIC 
# MAGIC Note that the experiment run information links directly to the remote tracking server, and no external link is necessary.

# COMMAND ----------

client.get_registered_model(registered_model_name).latest_versions

# COMMAND ----------

# MAGIC %md ## Referencing the *local* workspace Tracking Server

# COMMAND ----------

# DBTITLE 1,Register new model version
# Register a model from the local tracking server
mlflow.set_tracking_uri('databricks')
mlflow.register_model(
  model_uri=f'runs:/{local_run.info.run_id}/random-forest-model',
  name=registered_model_name)

# COMMAND ----------

# MAGIC %md ### Inspect results in remote model registry
# MAGIC 
# MAGIC Note that the run link takes us back to this local workspace.
# MAGIC 
# MAGIC Also note that the source DBFS location is a cached location, as this local workspace dbfs is not accessible from the remote workspace model registry.

# COMMAND ----------

client.get_registered_model(registered_model_name).latest_versions
