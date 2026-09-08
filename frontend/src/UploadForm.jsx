import { useRef, useState, useCallback } from 'react'

export default function UploadForm({
  previewUrl,
  onFileSelect,
  onAnalyze,
  onReset,
  loading,
  hasFile,
}) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const openFileDialog = () => inputRef.current?.click()

  const handleInputChange = (event) => {
    const selected = event.target.files?.[0]
    if (selected) onFileSelect(selected)
    // Clear the input so choosing the same file again still fires onChange.
    event.target.value = ''
  }

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault()
      setIsDragging(false)
      const dropped = event.dataTransfer.files?.[0]
      if (dropped) onFileSelect(dropped)
    },
    [onFileSelect],
  )

  const handleDragOver = (event) => {
    event.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => setIsDragging(false)

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openFileDialog()
    }
  }

  return (
    <section className="card upload-card">
      {!previewUrl ? (
        <div
          className={`dropzone${isDragging ? ' dropzone-active' : ''}`}
          onClick={openFileDialog}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onKeyDown={handleKeyDown}
          role="button"
          tabIndex={0}
          aria-label="Upload a photo of the skin lesion"
        >
          <svg className="dropzone-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M12 16V4M12 4l-4 4M12 4l4 4"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <p className="dropzone-title">Drag and drop a photo here</p>
          <p className="dropzone-subtitle">or click to browse, or use your camera on mobile</p>
        </div>
      ) : (
        <div className="preview-wrap">
          <img src={previewUrl} alt="Selected skin lesion" className="preview-image" />
          <button type="button" className="btn-text" onClick={onReset} disabled={loading}>
            Choose a different photo
          </button>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleInputChange}
        className="file-input-hidden"
      />

      <button
        type="button"
        className="btn-primary"
        onClick={onAnalyze}
        disabled={!hasFile || loading}
      >
        {loading ? (
          <>
            <span className="spinner" aria-hidden="true" />
            Analyzing…
          </>
        ) : (
          'Analyze'
        )}
      </button>
    </section>
  )
}
