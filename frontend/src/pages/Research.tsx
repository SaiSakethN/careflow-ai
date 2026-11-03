
import {
  Activity,
  BrainCircuit,
  Database,
  GitBranch,
  ShieldAlert,
  CheckCircle2,
} from 'lucide-react'

const pipeline = [
  {
    title: 'Data ingestion',
    description: 'Load and validate the CDC NHAMCS 2022 ED SAS dataset.',
  },
  {
    title: 'Target & features',
    description: 'Define admission outcomes and prepare pre-disposition features.',
  },
  {
    title: 'Hospital-grouped split',
    description: 'Separate hospitals across training, validation, and test sets.',
  },
  {
    title: 'Model training',
    description: 'Train and evaluate an XGBoost classification pipeline.',
  },
  {
    title: 'Model serving',
    description: 'Serve synthetic-input predictions through FastAPI and React.',
  },
]

const limitations = [
  'The dataset represents sampled emergency department visits from 2022 and may not reflect current populations or care practices.',
  'Model probabilities are not calibrated for individual clinical risk estimation.',
  'The 20% classification threshold is not a clinically validated decision threshold.',
  'Performance may vary across hospitals, patient groups, and time periods.',
  'Validation performance does not establish real-world clinical effectiveness.',
  'The held-out test set has not yet been evaluated.',
]

export default function Research() {
  return (
    <div className="research-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">RESEARCH DOCUMENTATION</span>
          <h1>Research & Limitations</h1>
          <p>
            Understand the dataset, methodology, reproducibility,
            and responsible-use boundaries of CareFlow AI.
          </p>
        </div>
      </div>

      <div className="performance-warning">
        <ShieldAlert size={22} />
        <div>
          <strong>Research prototype — not for clinical use</strong>
          <p>
            CareFlow AI is an educational machine-learning research
            project. It must not be used to make admission,
            discharge, treatment, or other patient-care decisions.
            Use synthetic demonstration inputs only.
          </p>
        </div>
      </div>

      <div className="research-grid">
        <section className="panel">
          <div className="panel-icon">
            <Database size={22} />
          </div>
          <h3>Dataset</h3>
          <p>
            2022 National Hospital Ambulatory Medical Care Survey
            (NHAMCS), Emergency Department public-use data.
          </p>
          <div className="research-facts">
            <div><span>Original visits</span><strong>16,025</strong></div>
            <div><span>Eligible visits</span><strong>15,919</strong></div>
            <div><span>Hospital admissions</span><strong>2,121</strong></div>
            <div><span>Sampled hospitals</span><strong>188</strong></div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-icon">
            <BrainCircuit size={22} />
          </div>
          <h3>Model design</h3>
          <p>
            XGBoost binary classifier using demographic,
            arrival, triage, and recorded vital-sign features.
          </p>
          <div className="research-facts">
            <div><span>Training visits</span><strong>11,390</strong></div>
            <div><span>Validation visits</span><strong>2,128</strong></div>
            <div><span>Held-out test visits</span><strong>2,401</strong></div>
            <div><span>Classification threshold</span><strong>0.20</strong></div>
          </div>
        </section>
      </div>

      <section className="panel research-pipeline">
        <div className="panel-icon">
          <GitBranch size={22} />
        </div>
        <h3>End-to-end research pipeline</h3>
        <div className="pipeline-steps">
          {pipeline.map((step, index) => (
            <div className="pipeline-step" key={step.title}>
              <div className="pipeline-number">{index + 1}</div>
              <div>
                <strong>{step.title}</strong>
                <p>{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel research-limitations">
        <div className="panel-icon">
          <ShieldAlert size={22} />
        </div>
        <h3>Known limitations</h3>
        <ul>
          {limitations.map((limitation) => (
            <li key={limitation}>
              <ShieldAlert size={16} />
              <span>{limitation}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel research-reproducibility">
        <div className="panel-icon">
          <Activity size={22} />
        </div>
        <h3>Reproducibility & transparency</h3>
        <p>
          The project includes source code for data ingestion,
          feature preprocessing, target construction, grouped
          splitting, model training, evaluation, and API serving.
          Reported performance values come from validation
          experiments, not from a production deployment.
        </p>
        <div className="research-checks">
          <span><CheckCircle2 size={16} /> Version-controlled code</span>
          <span><CheckCircle2 size={16} /> Automated backend tests</span>
          <span><CheckCircle2 size={16} /> Saved model metadata</span>
          <span><CheckCircle2 size={16} /> SHAP analysis</span>
        </div>
      </section>
    </div>
  )
}
