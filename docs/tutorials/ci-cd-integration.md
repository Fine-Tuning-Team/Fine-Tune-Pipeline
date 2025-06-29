# CI/CD Integration

This tutorial guides you through setting up automated fine-tuning pipelines using GitHub Actions and Jenkins, enabling continuous model training and deployment.

## Overview

CI/CD (Continuous Integration/Continuous Deployment) for ML models enables:

- **Automated Training**: Trigger training on data updates or schedule
- **Model Validation**: Automatic quality checks before deployment
- **Version Control**: Track model versions and configurations
- **Reproducible Pipelines**: Consistent training environments
- **Monitoring**: Track training metrics and model performance

## Prerequisites

- ✅ Git repository with Fine-Tune Pipeline
- ✅ GitHub account or Jenkins server access
- ✅ Cloud compute resources (GitHub Actions runners or cloud instances)
- ✅ API keys for Hugging Face, Weights & Biases, and OpenAI

## GitHub Actions Setup

### 1. Repository Configuration

First, set up your repository with the necessary secrets:

#### Repository Secrets

Go to **Settings > Secrets and Variables > Actions** and add:

```
HF_TOKEN              # Hugging Face API token
WANDB_TOKEN           # Weights & Biases API key
OPENAI_API_KEY        # OpenAI API key (for evaluation)
```

#### Directory Structure

```
.github/
├── workflows/
│   ├── fine-tune.yml           # Main training workflow
│   ├── evaluate.yml            # Evaluation workflow
│   ├── deploy.yml              # Model deployment
│   └── scheduled-training.yml  # Scheduled runs
├── scripts/
│   ├── setup-environment.sh    # Environment setup
│   ├── run-training.sh         # Training script
│   └── validate-model.sh       # Model validation
```

### 2. Main Training Workflow

Create `.github/workflows/fine-tune.yml`:

```yaml
name: Fine-Tune Model

on:
  push:
    branches: [ main ]
    paths: 
      - 'config.toml'
      - 'app/**'
  workflow_dispatch:
    inputs:
      config_file:
        description: 'Configuration file to use'
        required: false
        default: 'config.toml'
      experiment_name:
        description: 'Experiment name'
        required: false
        default: 'ci-cd-run'

jobs:
  fine-tune:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'

    - name: Install uv
      run: |
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo "$HOME/.cargo/bin" >> $GITHUB_PATH

    - name: Install dependencies
      run: |
        uv sync

    - name: Configure experiment
      run: |
        # Set experiment name
        EXPERIMENT_NAME="${{ github.event.inputs.experiment_name || 'ci-cd-auto' }}"
        echo "EXPERIMENT_NAME=$EXPERIMENT_NAME" >> $GITHUB_ENV
        
        # Update config with CI-specific settings
        sed -i "s/run_name_prefix = \"\"/run_name_prefix = \"$EXPERIMENT_NAME-\"/" config.toml
        sed -i 's/push_to_hub = false/push_to_hub = true/' config.toml

    - name: Run fine-tuning
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
        WANDB_TOKEN: ${{ secrets.WANDB_TOKEN }}
      run: |
        uv run app/finetuner.py

    - name: Run inference
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        uv run app/inferencer.py

    - name: Run evaluation
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        uv run app/evaluator.py

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: training-results
        path: |
          inferencer_output.jsonl
          evaluator_output_summary.json
          evaluator_output_detailed.xlsx
          models/fine_tuned/

    - name: Create release
      if: github.ref == 'refs/heads/main'
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: model-${{ env.EXPERIMENT_NAME }}-${{ github.sha }}
        release_name: Model Release ${{ env.EXPERIMENT_NAME }}
        body: |
          Automated model training completed.
          
          **Training Configuration:**
          - Config: ${{ github.event.inputs.config_file || 'config.toml' }}
          - Commit: ${{ github.sha }}
          - Experiment: ${{ env.EXPERIMENT_NAME }}
          
          **Results:**
          See attached artifacts for detailed results.
```

### 3. Scheduled Training Workflow

Create `.github/workflows/scheduled-training.yml`:

```yaml
name: Scheduled Model Training

on:
  schedule:
    # Run every Monday at 02:00 UTC
    - cron: '0 2 * * 1'
  workflow_dispatch:

jobs:
  scheduled-training:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up environment
      uses: ./.github/workflows/fine-tune.yml
      with:
        experiment_name: "scheduled-$(date +%Y%m%d)"

    - name: Check data freshness
      run: |
        # Add logic to check if new training data is available
        python scripts/check-data-updates.py

    - name: Notify on completion
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#ml-training'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 4. Model Validation Workflow

Create `.github/workflows/validate-model.yml`:

```yaml
name: Model Validation

on:
  workflow_run:
    workflows: ["Fine-Tune Model"]
    types: [completed]

jobs:
  validate:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Download artifacts
      uses: actions/download-artifact@v3
      with:
        name: training-results

    - name: Validate model performance
      run: |
        python scripts/validate-performance.py \
          --results evaluator_output_summary.json \
          --threshold 0.7

    - name: Security scan
      run: |
        # Scan model for potential security issues
        python scripts/security-scan.py models/fine_tuned/

    - name: Create validation report
      run: |
        python scripts/create-validation-report.py \
          --output validation-report.md

    - name: Comment on PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('validation-report.md', 'utf8');
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: report
          });
```

## Jenkins Setup

### 1. Jenkins Pipeline Configuration

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        HF_TOKEN = credentials('huggingface-token')
        WANDB_TOKEN = credentials('wandb-token')
        OPENAI_API_KEY = credentials('openai-key')
        EXPERIMENT_NAME = "jenkins-${BUILD_NUMBER}"
    }
    
    triggers {
        // Run daily at 2 AM
        cron('0 2 * * *')
        
        // Trigger on SCM changes
        pollSCM('H/15 * * * *')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                sh '''
                    # Install uv
                    curl -LsSf https://astral.sh/uv/install.sh | sh
                    export PATH="$HOME/.cargo/bin:$PATH"
                    
                    # Install dependencies
                    uv sync
                '''
            }
        }
        
        stage('Configure Training') {
            steps {
                script {
                    // Update configuration for CI environment
                    sh '''
                        sed -i "s/run_name_prefix = \"\"/run_name_prefix = \"${EXPERIMENT_NAME}-\"/" config.toml
                        sed -i 's/push_to_hub = false/push_to_hub = true/' config.toml
                        
                        # Reduce training for CI (optional)
                        sed -i 's/epochs = 3/epochs = 1/' config.toml
                    '''
                }
            }
        }
        
        stage('Fine-Tune Model') {
            steps {
                sh '''
                    export PATH="$HOME/.cargo/bin:$PATH"
                    uv run app/finetuner.py
                '''
            }
        }
        
        stage('Run Inference') {
            steps {
                sh '''
                    export PATH="$HOME/.cargo/bin:$PATH"
                    uv run app/inferencer.py
                '''
            }
        }
        
        stage('Evaluate Model') {
            steps {
                sh '''
                    export PATH="$HOME/.cargo/bin:$PATH"
                    uv run app/evaluator.py
                '''
            }
        }
        
        stage('Validate Results') {
            steps {
                script {
                    // Parse evaluation results
                    def evaluation = readJSON file: 'evaluator_output_summary.json'
                    def bleuScore = evaluation.overall_scores.bleu_score
                    
                    // Quality gate
                    if (bleuScore < 20) {
                        error("Model quality below threshold: BLEU = ${bleuScore}")
                    }
                    
                    echo "Model validation passed: BLEU = ${bleuScore}"
                }
            }
        }
        
        stage('Deploy Model') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    # Deploy to production environment
                    python scripts/deploy-model.py \
                        --model-path models/fine_tuned/ \
                        --environment production
                '''
            }
        }
    }
    
    post {
        always {
            // Archive artifacts
            archiveArtifacts artifacts: '''
                inferencer_output.jsonl,
                evaluator_output_summary.json,
                evaluator_output_detailed.xlsx
            ''', fingerprint: true
            
            // Cleanup
            sh 'rm -rf models/fine_tuned/'
        }
        
        success {
            // Notify success
            slackSend(
                channel: '#ml-training',
                color: 'good',
                message: "✅ Model training completed successfully: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
            )
        }
        
        failure {
            // Notify failure
            slackSend(
                channel: '#ml-training',
                color: 'danger',
                message: "❌ Model training failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
            )
        }
    }
}
```

### 2. Jenkins Configuration Script

Create `scripts/setup-jenkins.sh`:

```bash
#!/bin/bash

# Jenkins setup script
echo "Setting up Jenkins for Fine-Tune Pipeline..."

# Install required plugins
jenkins-cli install-plugin \
    pipeline-stage-view \
    workflow-aggregator \
    git \
    slack \
    build-timeout \
    credentials-binding

# Create credentials
jenkins-cli create-credentials-by-xml system::system::jenkins < credentials.xml

# Create job
jenkins-cli create-job fine-tune-pipeline < job-config.xml

echo "Jenkins setup complete!"
```

## Cloud Infrastructure

### 1. AWS Setup with GitHub Actions

Create `.github/workflows/aws-training.yml`:

```yaml
name: AWS GPU Training

on:
  workflow_dispatch:
    inputs:
      instance_type:
        description: 'EC2 instance type'
        required: true
        default: 'g4dn.xlarge'
        type: choice
        options:
        - g4dn.xlarge
        - g4dn.2xlarge
        - p3.2xlarge

jobs:
  train-on-aws:
    runs-on: ubuntu-latest
    
    steps:
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Launch training instance
      run: |
        # Launch EC2 instance
        INSTANCE_ID=$(aws ec2 run-instances \
          --image-id ami-0abcdef1234567890 \
          --instance-type ${{ github.event.inputs.instance_type }} \
          --key-name my-key-pair \
          --security-group-ids sg-903004f8 \
          --subnet-id subnet-6e7f829e \
          --user-data file://scripts/cloud-init.sh \
          --query 'Instances[0].InstanceId' \
          --output text)
        
        echo "INSTANCE_ID=$INSTANCE_ID" >> $GITHUB_ENV

    - name: Wait for instance ready
      run: |
        aws ec2 wait instance-running --instance-ids $INSTANCE_ID

    - name: Run training
      run: |
        # Get instance IP
        INSTANCE_IP=$(aws ec2 describe-instances \
          --instance-ids $INSTANCE_ID \
          --query 'Reservations[0].Instances[0].PublicIpAddress' \
          --output text)
        
        # Run training via SSH
        ssh -i ~/.ssh/my-key.pem ubuntu@$INSTANCE_IP \
          "cd /home/ubuntu/Fine-Tune-Pipeline && ./scripts/run-training.sh"

    - name: Download results
      run: |
        # Download training artifacts
        scp -i ~/.ssh/my-key.pem ubuntu@$INSTANCE_IP:/home/ubuntu/Fine-Tune-Pipeline/*.jsonl .

    - name: Terminate instance
      if: always()
      run: |
        aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

### 2. Google Cloud Setup

Create `scripts/gcp-training.sh`:

```bash
#!/bin/bash

# Google Cloud training script
PROJECT_ID="your-project-id"
ZONE="us-central1-a"
INSTANCE_NAME="fine-tune-worker"

# Create GPU instance
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --metadata-from-file startup-script=scripts/startup-script.sh

# Wait for startup
echo "Waiting for instance to be ready..."
sleep 120

# Run training
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --command="cd /opt/Fine-Tune-Pipeline && python -m app.finetuner"

# Download results
gcloud compute scp $INSTANCE_NAME:/opt/Fine-Tune-Pipeline/*.jsonl . \
    --project=$PROJECT_ID \
    --zone=$ZONE

# Cleanup
gcloud compute instances delete $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --quiet
```

## Monitoring and Alerting

### 1. Training Monitoring

Create `scripts/monitor-training.py`:

```python
import json
import requests
import time
from pathlib import Path

def monitor_training_progress():
    """Monitor training progress and send alerts."""
    
    # Check training logs
    log_file = Path("training.log")
    if not log_file.exists():
        return
    
    # Parse recent logs
    with open(log_file) as f:
        lines = f.readlines()
    
    # Extract metrics
    latest_loss = None
    for line in reversed(lines[-100:]):  # Last 100 lines
        if "train_loss" in line:
            latest_loss = extract_loss(line)
            break
    
    # Alert conditions
    if latest_loss and latest_loss > 2.0:
        send_alert(f"High training loss detected: {latest_loss}")
    
    # Check GPU utilization
    gpu_usage = get_gpu_utilization()
    if gpu_usage < 50:
        send_alert(f"Low GPU utilization: {gpu_usage}%")

def send_alert(message):
    """Send alert to Slack/Discord/Email."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if webhook_url:
        requests.post(webhook_url, json={"text": message})

if __name__ == "__main__":
    monitor_training_progress()
```

### 2. Model Performance Tracking

Create `scripts/track-performance.py`:

```python
import json
import sqlite3
from datetime import datetime

def track_model_performance():
    """Track model performance over time."""
    
    # Load evaluation results
    with open("evaluator_output_summary.json") as f:
        results = json.load(f)
    
    # Connect to database
    conn = sqlite3.connect("model_performance.db")
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            commit_hash TEXT,
            bleu_score REAL,
            rouge_score REAL,
            semantic_similarity REAL,
            config_hash TEXT
        )
    """)
    
    # Insert results
    cursor.execute("""
        INSERT INTO model_runs 
        (timestamp, commit_hash, bleu_score, rouge_score, semantic_similarity, config_hash)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        os.getenv("GITHUB_SHA", "unknown"),
        results["overall_scores"]["bleu_score"],
        results["overall_scores"]["rouge_1"],
        results["overall_scores"]["semantic_similarity"],
        hash_config("config.toml")
    ))
    
    conn.commit()
    conn.close()

def hash_config(config_path):
    """Generate hash of configuration for tracking."""
    import hashlib
    with open(config_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
```

## Best Practices

### 1. Configuration Management

```yaml
# Use different configs for different environments
configs/
├── development.toml     # Quick training for testing
├── staging.toml        # Full validation pipeline
└── production.toml     # Production training settings
```

### 2. Secret Management

```bash
# Use environment-specific secrets
# Development
export HF_TOKEN="hf_dev_token"

# Production  
export HF_TOKEN="hf_prod_token"

# Use secret management services
aws secretsmanager get-secret-value --secret-id "fine-tune/hf-token"
```

### 3. Resource Management

```yaml
# Auto-scaling for cost optimization
resource_limits:
  cpu: "4"
  memory: "16Gi"
  gpu: "1"
  
auto_scaling:
  min_replicas: 0
  max_replicas: 3
  target_gpu_utilization: 80
```

### 4. Model Versioning

```python
# Semantic versioning for models
def generate_model_version(config, performance):
    """Generate semantic version based on changes."""
    
    major = 1  # Breaking changes to model architecture
    minor = 0  # New features or significant improvements
    patch = 0  # Bug fixes or minor improvements
    
    # Auto-increment based on performance
    if performance["bleu_score"] > previous_best + 5:
        minor += 1
    elif performance["bleu_score"] > previous_best:
        patch += 1
    
    return f"v{major}.{minor}.{patch}"
```

## Troubleshooting

### Common CI/CD Issues

1. **Resource Limits**: Adjust instance types and memory limits
2. **Authentication**: Verify all API keys are properly set
3. **Timeouts**: Increase workflow timeouts for long training runs
4. **Dependencies**: Pin dependency versions for reproducibility

### Debugging Failed Runs

```bash
# Access logs from failed runs
gh run view <run-id> --log

# Debug locally with same environment
docker run --rm -it \
  -v $(pwd):/workspace \
  -w /workspace \
  pytorch/pytorch:latest \
  bash -c "pip install uv && uv sync && uv run app/finetuner.py"
```

With this CI/CD setup, your Fine-Tune Pipeline becomes a fully automated, monitored, and scalable machine learning system that can continuously improve your models while maintaining high quality standards.
