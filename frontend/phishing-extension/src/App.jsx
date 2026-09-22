/* global chrome */
import { useEffect, useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [email, setEmail] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [emailLoading, setEmailLoading] = useState(true)
  const [backendReady, setBackendReady] = useState(false)

  useEffect(() => {
    setEmailLoading(true)

    const listener = (request) => {
      if (request?.type === "EMAIL_DATA" && request.payload) {
        setEmail(request.payload)
        setEmailLoading(false)
      }
    }

    chrome.runtime.onMessage.addListener(listener)

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const activeTab = tabs[0]

      if (!activeTab?.id) {
        setEmailLoading(false)
        return
      }

      chrome.tabs.sendMessage(
        activeTab.id,
        { type: "GET_EMAIL_DATA" },
        (response) => {
          if (chrome.runtime.lastError || !response?.payload) {
            setEmail(null)
          } else {
            setEmail(response.payload)
          }
          setEmailLoading(false)
        }
      )
    })

    return () => {
      chrome.runtime.onMessage.removeListener(listener)
    }
  }, [])

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/health')
      .then(() => setBackendReady(true))
      .catch(() => setBackendReady(false))
  }, [])

  const handleScan = async () => {

    console.log("Scanning email:", email)

    if (!email) {
      setResult({
        label: "No email selected",
        explanation: "Open a Gmail message, refresh the page, and try again."
      })
      return
    }

    setLoading(true)
    setResult(null)

    try {

      const res = await axios.post(
        'http://127.0.0.1:5000/api/ml-detect',
        {
          subject: email.subject || "",
          sender: email.sender || "",
          body: email.body || ""
        }
      )

      console.log("ML response:", res.data)

      const {
        is_phishing,
        risk_level,
        risk_score,
        text_probability,
        url_probability,
        url_count,
        urls
      } = res.data

      setResult({

        label: is_phishing
          ? "Phishing Detected"
          : "Not Phishing",

        explanation: is_phishing
          ? `The ML system detected phishing indicators in this email.`
          : `The ML system did not detect strong phishing indicators.`,

        riskLevel: risk_level,

        riskScore: risk_score,

        textProbability: text_probability,

        urlProbability: url_probability,

        urlCount: url_count,

        urls: urls || []

      })

    } catch (err) {

      console.error("Detection error:", err)

      const isNetworkError = !err.response
      setResult({
        label: isNetworkError ? "Backend offline" : "Scan failed",
        explanation: isNetworkError
          ? "Start Flask with python app.py, keep that terminal open, then scan again."
          : err.response?.data?.error || "The detector could not process this email."
      })

    } finally {

      setLoading(false)

    }
  }

  return (
    <div className="container">
      <header className="app-header">
        <div>
          <p className="eyebrow">EMAIL SECURITY</p>
          <h1>Phishing Detector</h1>
        </div>
        <span className={`status ${backendReady ? 'status-ready' : 'status-offline'}`}>
          <span className="status-dot"></span>
          {backendReady ? 'Ready' : 'Offline'}
        </span>
      </header>

      {emailLoading ? (

        <div className="load-wrapper">

          <div className="loader"></div>

          <p className="load-text">
            Loading email...
          </p>

        </div>

      ) : email ? (

        <section className="card">
          <div className="section-label">MESSAGE PREVIEW</div>

          <div className="message-subject">
            <span className="field-label">SUBJECT</span>
            <h3>{email.subject || "No subject"}</h3>
          </div>

          <div className="sender-row">
            <span className="field-label">FROM</span>
            <span className="sender-value">{email.sender || "Unknown sender"}</span>
          </div>

          <p className="message-meta">
            {email.body.length.toLocaleString()} characters ready to scan
          </p>

          <p className="body-preview">{email.body.slice(0, 220)}{email.body.length > 220 ? "..." : ""}</p>
        </section>

      ) : (

        <p>
          Please open an email to begin scan.
        </p>

      )}

      {!emailLoading && email && (

        <button className="scan-button"
          onClick={handleScan}
          disabled={loading}
        >

          <span>{loading ? "Checking message..." : "Scan this email"}</span>

        </button>

      )}

      {loading && (

        <div className="load-wrapper">

          <div className="loader"></div>

          <p>
            Analysis in progress...
          </p>

        </div>

      )}

      {result && (

        <section className={`result-card ${result.riskLevel ? `risk-${result.riskLevel.toLowerCase()}` : 'result-error'}`}>

          <h2>
            {result.label}
          </h2>

          <p className="explanation">
            {result.explanation}
          </p>

          {result.riskScore !== undefined && (

            <div className="metrics-grid">

              <p className="metric">
                <strong>Risk Level:</strong>{" "}
                {result.riskLevel}
              </p>

              <p className="metric">
                <strong>Risk Score:</strong>{" "}
                {result.riskScore}%
              </p>

              <p className="metric">
                <strong>Text Model:</strong>{" "}
                {result.textProbability}%
              </p>

              <p className="metric">
                <strong>URL Model:</strong>{" "}
                {result.urlProbability}%
              </p>

              <p className="metric">
                <strong>URLs Found:</strong>{" "}
                {result.urlCount}
              </p>

              {result.urls.length > 0 && (
                  <div className="url-list">
                  <strong>Detected URLs:</strong>
                  <ul>
                    {result.urls.map((url) => (
                      <li key={url}>{url}</li>
                    ))}
                  </ul>
                </div>
              )}

            </div>

          )}

        </section>

      )}

    </div>
  )
}

export default App