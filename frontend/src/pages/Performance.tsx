
import {
  Activity,
  BarChart3,
  BrainCircuit,
  ShieldAlert,
  Target,
} from 'lucide-react'

const metrics = [
  {
    label: 'ROC-AUC',
    value: '0.8375',
    description: 'Discrimination across classification thresholds',
  },
  {
    label: 'PR-AUC',
    value: '0.3952',
    description: 'Precision-recall performance on validation data',
  },
  {
    label: 'Precision',
    value: '0.3333',
    description: 'At the selected 20% threshold',
  },
  {
    label: 'Recall',
    value: '0.6723',
    description: 'At the selected 20% threshold',
  },
  {
    label: 'F1-score',
    value: '0.4457',
    description: 'At the selected 20% threshold',
  },
]

export default function Performance() {
  return (
    <div className="performance-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">MODEL EVALUATION</span>
          <h1>Model Performance</h1>
          <p>
            Validation results from the XGBoost admission-prediction
            research pipeline.
          </p>
        </div>
      </div>

      <div className="performance-banner">
        <BarChart3 size={21} />
        <div>
          <strong>Hospital-disjoint validation</strong>
          <p>
            Metrics are from the validation split. The held-out test
            set has not yet been evaluated.
          </p>
        </div>
      </div>

      <div className="section-heading">
        <h2>Validation metrics</h2>
        <span>XGBoost · Original model</span>
      </div>

      <div className="performance-metrics">
        {metrics.map((metric) => (
          <div className="metric-card" key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
            <small>{metric.description}</small>
          </div>
        ))}
      </div>

      <div className="performance-details">
        <section className="panel">
          <div className="panel-icon">
            <Target size={22} />
          </div>
          <h3>Classification threshold</h3>
          <div className="threshold-value">20%</div>
          <p>
            Predictions at or above 0.20 are classified as
            admission-positive by this research model.
          </p>
          <p>
            This threshold is a model classification setting, not a
            clinically validated decision boundary.
          </p>
        </section>

        <section className="panel">
          <div className="panel-icon">
            <Activity size={22} />
          </div>
          <h3>Evaluation methodology</h3>
          <p>
            Visits were split by hospital to reduce information
            leakage between training and evaluation groups.
          </p>
          <p>
            Reported metrics are based on the validation partition,
            not the untouched held-out test partition.
          </p>
        </section>
      </div>

      <section className="panel shap-panel">
        <div className="shap-heading">
          <div>
            <div className="panel-icon">
              <BrainCircuit size={22} />
            </div>
            <h3>SHAP feature importance</h3>
            <p>
              Feature-attribution analysis for the trained
              XGBoost research model.
            </p>
          </div>
        </div>

        <img
          src="/figures/shap_feature_importance.png"
          alt="SHAP feature-importance chart for the XGBoost admission prediction model"
          className="shap-image"
          loading="lazy"
        />

        <p className="shap-caption">
          SHAP values describe model behavior and feature
          contributions. They do not establish clinical causation.
        </p>
      </section>

      <div className="performance-warning">
        <ShieldAlert size={22} />
        <div>
          <strong>Research-only interpretation</strong>
          <p>
            This model has not been clinically validated.
            Its probability outputs are not calibrated for
            patient-level clinical decision-making.
            Do not use these predictions to determine care,
            admission, or discharge.
          </p>
        </div>
      </div>
    </div>
  )
}
