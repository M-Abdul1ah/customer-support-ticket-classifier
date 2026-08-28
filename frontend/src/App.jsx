import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

const RESOLUTION_TIME = {
  high: 'within 2 hours',
  medium: 'within 24 hours',
  low: 'within 3 business days'
}

const GREETINGS = ['hi', 'hello', 'hey', 'hii', 'helo', 'yo', 'salam', 'assalamualaikum']

function generateTicketId() {
  const num = Math.floor(1000 + Math.random() * 9000)
  return `TICK-${num}`
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function App() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([
    {
      role: 'system',
      text: "Hi, I'm the support classifier. Type a message like a customer would, and I'll show you how it gets routed.",
      time: new Date()
    }
  ])
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSubmit = async (e) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages((prev) => [...prev, { role: 'user', text, time: new Date() }])
    setInput('')

    if (GREETINGS.includes(text.toLowerCase())) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: 'Hello! Tell me what you need help with, e.g. "I want to cancel my order" or "I forgot my password".',
          time: new Date()
        }
      ])
      return
    }

    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })

      if (!response.ok) throw new Error('Request failed')

      const data = await response.json()
      const ticketId = generateTicketId()
      const eta = RESOLUTION_TIME[data.priority] || 'as soon as possible'

      const replyText = data.requires_human
        ? `Ticket ${ticketId} created. This looks like a ${data.intent.replace(/_/g, ' ')} request. I'm not confident enough to handle it automatically, so it's being routed to a human agent in ${data.department}. Expected resolution: ${eta}.`
        : `Ticket ${ticketId} created. This looks like a ${data.intent.replace(/_/g, ' ')} request. I can handle this automatically, routing to ${data.department}. Expected resolution: ${eta}.`

      setMessages((prev) => [
        ...prev,
        { role: 'system', text: replyText, meta: data, ticketId, time: new Date() }
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'system', text: 'Could not reach the classifier. Is the API running on localhost:8000?', error: true, time: new Date() }
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <div className="header-dot" />
          <div>
            <h1>Customer Support</h1>
            <p className="subtitle">Live ticket classification demo</p>
          </div>
        </div>

        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`bubble-row ${m.role}`}>
              <div className={`bubble ${m.role} ${m.error ? 'error' : ''}`}>
                {m.ticketId && <div className="ticket-tag">{m.ticketId}</div>}
                <div>{m.text}</div>
                {m.meta && (
                  <div className="meta">
                    <span className={`badge priority-${m.meta.priority}`}>{m.meta.priority} priority</span>
                    <span className="badge">{(m.meta.confidence * 100).toFixed(0)}% confidence</span>
                    <span className={`badge status-${m.meta.status?.toLowerCase()}`}>{m.meta.status}</span>
                  </div>
                )}
                <div className="timestamp">{formatTime(m.time)}</div>
              </div>
            </div>
          ))}
          {loading && (
            <div className="bubble-row system">
              <div className="bubble system typing">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={handleSubmit} className="message-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()} aria-label="Send message">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M3 11.5L21 3L14 21L11 13L3 11.5Z" stroke="white" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"/>
            </svg>
          </button>
        </form>
      </div>
    </div>
  )
}

export default App