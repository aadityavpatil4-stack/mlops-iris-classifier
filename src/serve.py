import json
import joblib
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_PATH = "models/final_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


class PredictionHandler(BaseHTTPRequestHandler):

    def send_json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "healthy"})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/predict":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))

            missing = [
                feature for feature in FEATURE_COLUMNS
                if feature not in data
            ]

            if missing:
                self.send_json(400, {
                    "error": "Missing required features",
                    "missing_features": missing
                })
                return

            values = [[float(data[col]) for col in FEATURE_COLUMNS]]

            prediction = model.predict(values)[0]
            probabilities = model.predict_proba(values)[0]

            species = encoder.inverse_transform([prediction])[0]

            self.send_json(200, {
                "prediction": species,
                "probabilities": probabilities.tolist()
            })

        except (ValueError, TypeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Invalid input"})
        except Exception:
            self.send_json(500, {"error": "Prediction failed"})


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), PredictionHandler)
    print("Iris prediction server running on port 8000...")
    server.serve_forever()
