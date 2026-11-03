
export interface AdmissionRequest {
  age: number
  sex: number
  residence: number | null
  triage_immediacy: number | null
  temperature_f: number | null
  pulse: number | null
  respiratory_rate: number | null
  systolic_bp: number | null
  diastolic_bp: number | null
  oxygen_saturation: number | null
  pain_scale: number | null
  visit_month: number
  visit_day: number
  arrival_time: string
}

export interface AdmissionResponse {
  admission_probability: number
  predicted_admission: boolean
  threshold: number
  model_version: string
  research_only: boolean
  disclaimer: string
}
