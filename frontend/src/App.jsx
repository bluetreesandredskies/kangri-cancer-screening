import { useState, useCallback, useEffect } from 'react'
import UploadForm from './UploadForm.jsx'
import ResultCard from './ResultCard.jsx'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // Revoke the object URL whenever it's replaced or the app unmounts, so we
  // don't leak memory across repeated selections.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const handleFileSelect = useCallback((selected) => {
    if (!selected) return
    if (!selected.type.startsWith('image/')) {
      setResult(null)
      setError("That file doesn't look like an image. Please choose a photo of the skin lesion.")
      return
    }
    setError(null)
    setResult(null)
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
  }, [])

  const handleReset = useCallback(() => {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        let message = `The server responded with an error (${response.status}).`
        try {
          const body = await response.json()
          if (body && body.detail) message = body.detail
        } catch {
          // response body wasn't JSON — fall back to the generic message above
        }
        throw new Error(message)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      if (err instanceof TypeError) {
        // fetch() throws a TypeError for network-level failures: server down,
        // wrong VITE_API_URL, CORS blocked, DNS failure, etc.
        setError("Couldn't reach the server. Is the backend running?")
      } else {
        setError(err.message || 'Something went wrong while analyzing the image.')
      }
    } finally {
      setLoading(false)
    }
  }, [file])

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-inner">
          <div className="hero-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="6.5" stroke="currentColor" strokeWidth="1.6" />
              <path d="M20 20l-3.8-3.8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              <path d="M8.2 11a2.8 2.8 0 0 1 2.8-2.8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </div>
          <h1>Paakzir</h1>
          <p className="hero-tagline">
            Upload a photo of a skin lesion and get an instant, AI-assisted risk
            screening. Paakzir flags patterns worth a closer look, so you know
            when it's time to see a dermatologist.
          </p>
        </div>
      </header>

      <div className="disclaimer-banner" role="note">
        <span className="disclaimer-icon" aria-hidden="true">i</span>
        Screening aid only — not a medical diagnosis.
      </div>

      <main className="main">
        <UploadForm
          previewUrl={previewUrl}
          onFileSelect={handleFileSelect}
          onAnalyze={handleAnalyze}
          onReset={handleReset}
          loading={loading}
          hasFile={!!file}
        />

        {(result || error) && <ResultCard result={result} error={error} />}
      </main>

      <footer className="footer">
        <p>Built by Aneek Debnath, Manu Bhagat, Jeetu Kumar and Sakshi Prajapati</p>
      </footer>
    </div>
  )
}
