# FraudGuard AI - n8n Agent Workflow

Ce dossier contient une base de workflow `n8n` pour transformer la notification en **agent IA orchestré**.

## Objectif

Le backend FastAPI ne contacte plus Groq directement pour la notification utilisateur.

Le flux attendu devient :

1. `Streamlit` appelle `FastAPI /predict`
2. `Streamlit` appelle `FastAPI /notify_user`
3. `FastAPI` génère `confirm_url` / `reject_url`
4. `FastAPI` envoie un payload JSON au webhook `n8n`
5. `n8n` :
   - prépare le contexte
   - appelle `Groq API`
   - génère le sujet + message email
   - envoie l'email
   - renvoie un statut à FastAPI
6. l'utilisateur clique sur le lien de confirmation
7. `FastAPI /feedback-action` écrit le feedback dans `prod_data.csv`

## Fichiers

- `fraudguard_notify_workflow.json`
  Template de workflow n8n à importer.
- `notify_user_payload.example.json`
  Exemple du payload que FastAPI envoie au webhook n8n.

## Variables d'environnement côté n8n

Le service `n8n` dans `deployment/docker/compose.yaml` attend notamment :

- `GROQ_API_KEY`
- `FRAUD_GROQ_MODEL`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- `SMTP_USE_TLS`
- `FRAUD_PUBLIC_BASE_URL`

## Webhook attendu

FastAPI appelle ce webhook :

`POST /webhook/fraudguard-notify`

Le payload contient :

- `prediction_id`
- `customer_email`
- `amount`
- `prediction`
- `predicted_label`
- `fraud_probability`
- `explanatory_factors`
- `confirm_url`
- `reject_url`
- `email_subject`
- `customer_message_preview`
- `recommended_action`
- `analysis_timestamp`
- `analysis_context`

## Logique du workflow n8n

### 1. Webhook Trigger

Reçoit la requête JSON de FastAPI.

### 2. Set / Prepare Context

Construit :

- les données transactionnelles
- le prompt système
- le prompt utilisateur
- le corps d'appel Groq compatible OpenAI

### 3. HTTP Request -> Groq

Appelle :

`POST https://api.groq.com/openai/v1/chat/completions`

Headers :

- `Authorization: Bearer {{$env.GROQ_API_KEY}}`
- `Content-Type: application/json`

Body JSON :

- `model: {{$env.FRAUD_GROQ_MODEL || "openai/gpt-oss-20b"}}`
- `messages`
- `temperature`
- `response_format` si souhaité

### 4. Parse JSON

Analyse la réponse Groq et extrait :

- `email_subject`
- `customer_message`
- `recommended_action`

### 5. Email Node

Envoie l'email au client avec :

- destinataire = `customer_email`
- sujet = `email_subject`
- corps = message Groq + liens `confirm_url` / `reject_url`

### 6. Respond to Webhook

Retourne à FastAPI un JSON du type :

```json
{
  "status": "accepted",
  "workflow_status": "email_sent",
  "email_subject": "Vérification de votre transaction",
  "provider": "n8n"
}
```

## Import

1. Démarrer n8n
2. Ouvrir `http://localhost:5678`
3. Importer `fraudguard_notify_workflow.json`
4. Configurer les credentials SMTP ou le node email utilisé
5. Vérifier que le chemin du webhook est bien `fraudguard-notify`
6. Activer le workflow

## Remarque

Le fichier JSON fourni est un **template d'import** pour accélérer ton setup.
Selon ta version de n8n et ton node email préféré, il peut être nécessaire d'ajuster :

- les credentials du node email
- le format exact du parse JSON
- le modèle Groq
