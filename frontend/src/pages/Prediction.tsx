
import { useState, type FormEvent } from 'react'
import { Activity, AlertCircle, ArrowRight, Loader2 } from 'lucide-react'
import { predictAdmission } from '../services/api'
import type {
  AdmissionRequest,
  AdmissionResponse,
} from '../types/prediction'

const example: AdmissionRequest = {
  age: 65,
  sex: 1,
  residence: 1,
  triage_immediacy: 3,
  temperature_f: 98.6,
  pulse: 92,
  respiratory_rate: 20,
  systolic_bp: 140,
  diastolic_bp: 85,
  oxygen_saturation: 97,
  pain_scale: 4,
  visit_month: 11,
  visit_day: 1,
  arrival_time: '1430',
}

type NumericField = Exclude<keyof AdmissionRequest, 'arrival_time'>

const fields: {
  key: NumericField
  label: string
  min: number
  max: number
  step?: number
}[] = [
  { key: 'age', label: 'Age (years)', min: 0, max: 120 },
  { key: 'sex', label: 'Sex code (1–2)', min: 1, max: 2 },
  { key: 'residence', label: 'Residence code (1–4)', min: 1, max: 4 },
  { key: 'triage_immediacy', label: 'Triage immediacy (1–5)', min: 1, max: 5 },
  { key: 'temperature_f', label: 'Temperature (°F)', min: 85, max: 110, step: 0.1 },
  { key: 'pulse', label: 'Pulse (bpm)', min: 0, max: 240 },
  { key: 'respiratory_rate', label: 'Respiratory rate', min: 0, max: 150 },
  { key: 'systolic_bp', label: 'Systolic BP (mmHg)', min: 0, max: 289 },
  { key: 'diastolic_bp', label: 'Diastolic BP (mmHg)', min: 0, max: 190 },
  { key: 'oxygen_saturation', label: 'Oxygen saturation (%)', min: 0, max: 100 },
  { key: 'pain_scale', label: 'Pain scale (0–10)', min: 0, max: 10 },
  { key: 'visit_month', label: 'Visit month (1–12)', min: 1, max: 12 },
  { key: 'visit_day', label: 'Visit day (1–7)', min: 1, max: 7 },
]

export default function Prediction() {
  const [form, setForm] = useState<AdmissionRequest>({ ...example })
  const [result, setResult] = useState<AdmissionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function updateNumber(key: NumericField, value: string) {
    setForm((previous) => ({
      ...previous,
      [key]: value === '' ? null : Number(value),
    }))
    setResult(null)
    setError('')
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const prediction = await predictAdmission(form)
      setResult(prediction)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="prediction-page">
      <div className="page-heading">
        <span className="eyebrow">MODEL INFERENCE WORKSPACE</span>
        <h1>Admission Prediction</h1>
        <p>
          Explore model predictions using synthetic demonstration data.
          Do not enter real patient information.
        </p>
      </div>

      <div className="prediction-layout">
        <section className="panel">
          <h2>Demonstration visit details</h2>
          <p>Modify the example values to explore model behavior.</p>

          <form onSubmit={submit}>
            <div className="prediction-fields">
              {fields.map((field) => (
                <label className="field-group" key={field.key}>
                  <span>{field.label}</span>
                  <input
                    type="number"
                    min={field.min}
                    max={field.max}
                    step={field.step ?? 1}
                    required={
                      field.key === 'age' ||
                      field.key === 'sex' ||
                      field.key === 'visit_month' ||
                      field.key === 'visit_day'
                    }
                    value={form[field.key] ?? ''}
                    onChange={(event) =>
                      updateNumber(field.key, event.target.value)
                    }
                  />
                </label>
              ))}

              <label className="field-group">
                <span>Arrival time (HHMM)</span>
                <input
                  type="text"
                  value={form.arrival_time}
                  pattern="([01][0-9]|2[0-3])[0-5][0-9]"
                  maxLength={4}
                  required
                  onChange={(event) => {
                    setForm((previous) => ({
                      ...previous,
                      arrival_time: event.target.value,
                    }))
                    setResult(null)
                    setError('')
                  }}
                />
              </label>
            </div>

            <div className="prediction-actions">
              <button
                type="submit"
                className="primary-button"
                disabled={loading}
              >
                {loading ? (
                  <Loader2 size={17} className="spin" />
                ) : (
                  <Activity size={17} />
                )}
                {loading ? 'Calculating...' : 'Generate Prediction'}
                {!loading && <ArrowRight size={17} />}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => {
                  setForm({ ...example })
                  setResult(null)
                  setError('')
                }}
              >
                Reset example
              </button>
            </div>
          </form>
        </section>

        <section className="panel prediction-result">
          <h2>Model output</h2>
          <p>XGBoost admission-risk research model · v1.0.0</p>

          {error && (
            <div className="prediction-error" role="alert">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

          {!result && !error && (
            <div className="result-empty">
              <Activity size={40} />
              <h3>Awaiting prediction</h3>
              <p>
                Submit the demonstration form to generate a model estimate.
              </p>
            </div>
          )}

          {result && (
            <div className="result-content" aria-live="polite">
              <span className="result-label">Estimated admission probability</span>
              <strong className="result-value">
                {(result.admission_probability * 100).toFixed(2)}%
              </strong>

              <div
                className="risk-track"
                role="progressbar"
                aria-label="Estimated admission probability"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={result.admission_probability * 100}
              >
                <div
                  className="risk-fill"
                  style={{
                    width: `${result.admission_probability * 100}%`,
                  }}
                />
              </div>

              <p>
                Decision threshold: {(result.threshold * 100).toFixed(0)}%
              </p>

              <div className="result-status">
                {result.predicted_admission
                  ? 'Above selected admission threshold'
                  : 'Below selected admission threshold'}
              </div>

              <div className="research-disclaimer">
                <AlertCircle size={18} />
                <span>{result.disclaimer}</span>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
