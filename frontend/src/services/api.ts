
import type {
  AdmissionRequest,
  AdmissionResponse,
} from '../types/prediction'

export async function predictAdmission(
  data: AdmissionRequest,
): Promise<AdmissionResponse> {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    throw new Error(
      `Prediction request failed (HTTP ${response.status})`,
    )
  }

  return (await response.json()) as AdmissionResponse
}
