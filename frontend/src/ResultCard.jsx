const RISK_CONFIG = {
  none: { label: 'No lesion detected', className: 'badge-none' },
  low: { label: 'Low risk', className: 'badge-low' },
  moderate: { label: 'Moderate risk', className: 'badge-moderate' },
  high: { label: 'High risk', className: 'badge-high' },
}

function formatClassName(name) {
  if (!name) return ''
  return name
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

export default function ResultCard({ result, error }) {
  if (error) {
    return (
      <section className="card result-card">
        <div className="error-box">
          <svg className="error-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
            <path d="M12 8v5M12 16h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          <p>{error}</p>
        </div>
      </section>
    )
  }

  if (!result) return null

  const risk = RISK_CONFIG[result.risk_level] || RISK_CONFIG.moderate

  return (
    <section className="card result-card">
      <div className={`badge ${risk.className}`}>{risk.label}</div>

      <p className="result-meta">
        Detected pattern: <strong>{formatClassName(result.predicted_class)}</strong>
        {' · '}
        {result.confidence}% confidence
      </p>

      <p className="recommendation">{result.recommendation}</p>

      {result.safety_escalated && result.escalation_note && (
        <div className="escalation-note">
          <svg className="escalation-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 3l9 16H3l9-16z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
            <path d="M12 10v4M12 17h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          <p>{result.escalation_note}</p>
        </div>
      )}

      <p className="disclaimer-text">{result.disclaimer}</p>
    </section>
  )
}
