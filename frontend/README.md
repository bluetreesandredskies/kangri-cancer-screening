# Paakzir — Frontend

A Vite + React frontend for the Paakzir skin-lesion screening demo. It uploads
a photo to a backend REST API and displays the returned risk assessment.

## Setup

From inside the `frontend/` folder:

```bash
npm install
```

Create your local environment file from the example:

```bash
cp .env.example .env
```

(On Windows PowerShell, use `Copy-Item .env.example .env` instead.)

By default `.env` points the app at `http://localhost:8000`. Edit
`VITE_API_URL` in `.env` if your backend runs somewhere else.

Start the dev server:

```bash
npm run dev
```

This prints a local URL, typically `http://localhost:5173`. Open it in your
browser. Make sure the backend (`uvicorn backend.main:app --reload --port
8000`, run from the repo root) is running too, or the "Analyze" button will
show a "Couldn't reach the server" error.
