
import { useState } from 'react'
import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  ClipboardList,
  HeartPulse,
  LayoutDashboard,
  Menu,
  ShieldAlert,
  X,
} from 'lucide-react'

import Prediction from './pages/Prediction'
import './App.css'

type Page = 'overview' | 'prediction' | 'performance' | 'about'

const navigation = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'prediction', label: 'Admission Prediction', icon: ClipboardList },
  { id: 'performance', label: 'Model Performance', icon: BarChart3 },
  { id: 'about', label: 'Research & Limitations', icon: ShieldAlert },
] as const

const pageTitles: Record<Page, string> = {
  overview: 'Research Overview',
  prediction: 'Admission Prediction',
  performance: 'Model Performance',
  about: 'Research & Limitations',
}

function App() {
  const [page, setPage] = useState<Page>('overview')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  function navigate(nextPage: Page) {
    setPage(nextPage)
    setSidebarOpen(false)
  }

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button
          className="mobile-overlay"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-icon">
            <HeartPulse size={25} />
          </div>

          <div>
            <strong>CareFlow AI</strong>
            <span>Research Platform</span>
          </div>

          <button
            className="mobile-close"
            aria-label="Close menu"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        <div className="sidebar-section-label">WORKSPACE</div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? 'active' : ''}`}
              onClick={() => navigate(id)}
              aria-current={page === id ? 'page' : undefined}
            >
              <Icon size={19} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="research-notice">
            <ShieldAlert size={19} />

            <div>
              <strong>Research Prototype</strong>
              <p>Not intended for clinical decision-making.</p>
            </div>
          </div>

          <span className="sidebar-version">
            CareFlow AI · v1.0.0
          </span>
        </div>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="menu-button"
              aria-label="Open navigation"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu size={22} />
            </button>

            <div className="breadcrumb">
              Workspace <span>/</span>{' '}
              <strong>{pageTitles[page]}</strong>
            </div>
          </div>

          <div className="topbar-right">
            <span className="environment-badge">
              <span className="status-dot" />
              Research Environment
            </span>
          </div>
        </header>

        <main className="content">
          {page === 'overview' && (
            <>
              <div className="page-heading">
                <div>
                  <span className="eyebrow">
                    EMERGENCY DEPARTMENT ANALYTICS
                  </span>

                  <h1>Research Overview</h1>

                  <p>
                    Explore emergency department admission prediction
                    using a reproducible machine-learning pipeline.
                  </p>
                </div>
              </div>

              <section className="hero-card">
                <div className="hero-content">
                  <span className="hero-tag">
                    <BrainCircuit size={15} />
                    MACHINE LEARNING RESEARCH
                  </span>

                  <h2>Understanding hospital admission risk</h2>

                  <p>
                    A research platform built around the 2022 CDC NHAMCS
                    emergency department dataset and an XGBoost model.
                  </p>

                  <button
                    className="primary-button"
                    onClick={() => navigate('prediction')}
                  >
                    Explore Prediction
                    <ArrowRight size={18} />
                  </button>
                </div>

                <div className="hero-visual" aria-hidden="true">
                  <Activity size={105} strokeWidth={1.1} />
                </div>
              </section>

              <div className="section-heading">
                <h2>Dataset & model snapshot</h2>
                <span>2022 NHAMCS ED research dataset</span>
              </div>

              <div className="metrics-grid">
                <div className="metric-card">
                  <span>Eligible ED visits</span>
                  <strong>15,919</strong>
                  <small>
                    After target eligibility filtering
                  </small>
                </div>

                <div className="metric-card">
                  <span>Hospital admissions</span>
                  <strong>2,121</strong>
                  <small>Observed admission outcomes</small>
                </div>

                <div className="metric-card">
                  <span>Validation ROC-AUC</span>
                  <strong>0.8375</strong>
                  <small>
                    Hospital-disjoint validation split
                  </small>
                </div>

                <div className="metric-card">
                  <span>Validation PR-AUC</span>
                  <strong>0.3952</strong>
                  <small>Original XGBoost model</small>
                </div>
              </div>

              <div className="info-grid">
                <section className="panel">
                  <div className="panel-icon">
                    <BrainCircuit size={22} />
                  </div>

                  <h3>Admission-risk research</h3>

                  <p>
                    Explore predictions using synthetic demonstration
                    inputs and a saved machine-learning pipeline.
                  </p>

                  <button
                    className="text-button"
                    onClick={() => navigate('prediction')}
                  >
                    Open workspace
                    <ArrowRight size={16} />
                  </button>
                </section>

                <section className="panel">
                  <div className="panel-icon">
                    <BarChart3 size={22} />
                  </div>

                  <h3>Model evaluation</h3>

                  <p>
                    Review validation metrics, threshold selection,
                    and limitations of the current research model.
                  </p>

                  <button
                    className="text-button"
                    onClick={() => navigate('performance')}
                  >
                    View performance
                    <ArrowRight size={16} />
                  </button>
                </section>
              </div>
            </>
          )}

          {page === 'prediction' && <Prediction />}

          {(page === 'performance' || page === 'about') && (
            <section className="placeholder-page">
              <div className="placeholder-icon">
                {page === 'performance' ? (
                  <BarChart3 size={30} />
                ) : (
                  <ShieldAlert size={30} />
                )}
              </div>

              <h1>{pageTitles[page]}</h1>

              <p>
                This workspace is being implemented in the next stage
                of the CareFlow AI frontend.
              </p>

              <button
                className="secondary-button"
                onClick={() => navigate('overview')}
              >
                Return to overview
              </button>
            </section>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
