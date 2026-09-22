import { useEffect, useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [email, setEmail] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [emailLoading, setEmailLoading] = useState(true);

  useEffect(() => {
    setEmailLoading(true);

    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.type === "EMAIL_DATA") {
        setEmail(request.payload);
        setEmailLoading(false);
      }
      console.log("Extracted email:" + request.payload);
    });
  }, []);

  const testEmail = {
    subject: "Your order has been executed",
    sender: "noreply@robinhood.com",
    body: "Your order to sell $30.00 of AMD through your individual account was executed on April 9, 2025 at 3:51 PM ET. You received $30.00 for 0.30832 shares, at an average price of $97.29 per share. Funds available from this trade will be reflected in your withdrawable cash on Apr. 10, 2025."
  }

const handleScan = async () => {
    console.log(email)
    if (!email || !email.subject || !email.sender || !email.body) {
    setResult({
      label: "Error",
      explanation: "Missing required email fields (subject, sender, body). Please ensure the email is fully captured."
    });
    return;
  }
  setLoading(true);


    try {
      const res = await axios.post('http://127.0.0.1:5000/api/llm-query', {
      subject: email.subject,
      sender: email.sender,
      body: email.body
    });
    const { is_phishing, explanation } = res.data;
    setResult({
      label: is_phishing ? "⚠️ Phishing Detected" : "✅ Not Phishing",
      explanation
    });
    console.log(res.data)

    } catch (err) {
      console.error(err)
      setResult({
        label: "Error",
        explanation: err.response?.data?.error || err.message
      })
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className = "container">
      <h1>OpenAI Phishing Detector</h1>

      {emailLoading ? (
        <div className="load-wrapper">
          <div className="loader"></div>
          <p className="load-text">Loading email...</p>
        </div>
      ) : email ? (
        <div className="card">
          <h3>{email.subject}</h3>
          <p>From: {email.sender}</p>
          <p>{email.body}</p>
        </div>
      ) : (
        <p>Please open an email to begin scan.</p>
      )}

      {!emailLoading && email && (
        <button onClick={handleScan}>
          Scan Email
        </button> 
      )}

      {loading && <div className="loader"></div>}
      {result && (
        <div className={"result-card"}>
          <h2>{result.label}</h2>
          <p>{result.explanation}</p>
        </div>
      )}
    </div>
  )
}

export default App
