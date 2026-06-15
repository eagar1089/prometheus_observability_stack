# Digital Memory Jar - Backend

FastAPI-powered backend for the Digital Memory Jar platform. Handles authentication, memory management, AI analysis, and NLP processing.

## Quick Start

### Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running Locally

```bash
# Set required environment variables
export MONGO_URI="your_mongodb_uri"
export HF_API_TOKEN="your_huggingface_token"

# Start the development server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

Prometheus metrics are available at `http://localhost:8000/metrics`.

---

## Environment Configuration

<details>
<summary><b>Core Database & Authentication</b></summary>

- `MONGO_URI` - MongoDB connection string
- Firebase credentials for token verification (see `auth_deps.py`)

</details>

<details>
<summary><b>Spotify Integration</b></summary>

- `SPOTIFY_CLIENT_ID` - Spotify application client ID
- `SPOTIFY_CLIENT_SECRET` - Spotify application client secret

Used for generating mood-aware music recommendations based on emotional context of memories.

</details>

<details>
<summary><b>Hugging Face & NLP Models</b></summary>

- `HF_API_TOKEN` - Hugging Face API token for inference

**NLP Models Used:**
- `KEYBERT_MODEL` - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (keyword extraction)
- `NER_MODEL` - `xx_ent_wiki_sm` (entity recognition)
- `EMBEDDING_MODEL` - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (embeddings)
- `TOPIC_LABELS` - Comma-separated candidate labels for topic classification

**Supported Languages:** Marathi, English, Hindi

</details>

---

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `GET /auth/verify` - Verify Firebase token

### Memory Management (CRUD)
- `POST /memories/` - Create new memory
- `GET /memories/` - List user's memories
- `GET /memories/{id}` - Get specific memory
- `PUT /memories/{id}` - Update memory
- `DELETE /memories/{id}` - Delete memory

### AI Features
<details>
<summary><b>View AI Endpoints</b></summary>

- `GET /ai/weekly-reflection` - Generate weekly summary based on user's memories
- `GET /ai/mood-anomaly` - Detect anomaly signals using recent vs baseline mood patterns
- `POST /ai/companion-chat` - Contextual chat assistant grounded in user's memories
  - Body: `{ "question": "..." }`

</details>

### Dashboard & Analytics
- `GET /dashboard/` - User dashboard overview
- `GET /dashboard/mood-trends` - Historical mood data

### Music Recommendations
- `GET /spotify/recommendations` - Get mood-aware Spotify suggestions

### Monitoring
- `GET /metrics` - Prometheus metrics for Grafana dashboards and alerting

---

## NLP Pipeline

<details>
<summary><b>Emotion Scoring</b></summary>

Uses Hugging Face Inference API for multilingual emotion classification.
- **Environment:** `HF_API_TOKEN`
- **Supported languages:** Marathi, English, Hindi
- **Output:** Emotion labels with confidence scores

</details>

<details>
<summary><b>Keyword Extraction</b></summary>

KeyBERT-based extraction using multilingual sentence transformers.
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Supported languages:** Marathi, English, Hindi
- **Output:** List of relevant keywords from memory text

</details>

<details>
<summary><b>Topic Categorization</b></summary>

Zero-shot topic classification using Hugging Face models.
- **Requirement:** `TOPIC_LABELS` environment variable (comma-separated labels)
- **Supported languages:** Marathi, English, Hindi
- **Output:** Primary topic and confidence score

</details>

<details>
<summary><b>Entity Recognition (NER)</b></summary>

Multilingual named entity recognition with spaCy fallback.
- **Model:** `xx_ent_wiki_sm`
- **Supported languages:** Marathi, English, Hindi
- **Output:** Named entities (persons, locations, organizations, etc.)

</details>

<details>
<summary><b>Embedding Generation</b></summary>

Semantic embeddings for memory similarity and search.
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Output:** 384-dimensional vectors for semantic matching

</details>

---

## Project Structure

```
backend/
├── main.py                 # FastAPI app entry point
├── auth_deps.py           # Firebase authentication middleware
├── connection.py          # MongoDB connection management
├── crud.py                # Database operations
├── schemas.py              # Pydantic schemas for validation
├── nlp_processor.py       # NLP pipeline orchestration
├── text_preprocessor.py   # Text cleaning and preprocessing
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment configuration
├── routers/               # API route handlers
│   ├── auth.py
│   ├── memories.py
│   ├── dashboard.py
│   ├── ai_features.py
│   ├── emotion_analyzer.py
│   └── spotify.py
├── nlp/                   # NLP utilities
└── tests/
    └── test_backend_features.py
```

---

## Development Notes

- **Background Jobs:** APScheduler handles asynchronous NLP processing of memories
- **Database:** MongoDB Atlas for persistence
- **Authentication:** Firebase Admin SDK for token verification
- **Deployment:** Hugging Face Spaces (automated via GitHub Actions)
- **Monitoring:** Prometheus scrape endpoint plus Grafana dashboards via Docker Compose

---

## Troubleshooting

<details>
<summary><b>Common Issues</b></summary>

**MongoDB Connection Error**
- Verify `MONGO_URI` is set correctly
- Check network access in MongoDB Atlas

**Hugging Face API Errors**
- Ensure `HF_API_TOKEN` has sufficient quota
- Check model availability in your region

**Firebase Authentication Fails**
- Verify Firebase credentials are properly configured
- Check token expiration and validity

</details>